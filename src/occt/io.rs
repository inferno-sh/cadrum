//! I/O helpers for `Solid`. Exposed via `impl SolidStruct for Solid` in
//! `super::solid` (e.g. `Solid::read_step`, `Solid::write_step`, `Solid::mesh`).

use super::compound::CompoundShape;
use super::ffi;
use super::ffi::{RustReader, RustWriter};
use super::solid::Solid;
use crate::common::error::Error;
use crate::common::mesh::Mesh;
use crate::traits::Tessellation;
use glam::DMat4;
use std::io::{Read, Write};

#[cfg(feature = "color")]
use crate::common::color::Color;

/// STEP entity types for which OCCT exposes a direct topological transfer result.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StepEntityKind {
	/// `MANIFOLD_SOLID_BREP` exactly, excluding its more specific subtypes.
	ManifoldSolidBrep,
	/// `BREP_WITH_VOIDS` exactly.
	BrepWithVoids,
	/// `FACETED_BREP` exactly.
	FacetedBrep,
	/// Complex `FACETED_BREP_AND_BREP_WITH_VOIDS`.
	FacetedBrepWithVoids,
	/// `ADVANCED_FACE` exactly.
	AdvancedFace,
}

/// The kernel topology category expected from a STEP entity transfer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StepTopologyKind {
	Solid,
	Face,
}

/// Why a source entity has no exact identity in the returned topology.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StepTopologyUnboundReason {
	/// OCCT did not produce the entity's expected solid or face result.
	NoExpectedTransferResult,
	/// Import recovery rebuilt or discarded the transfer result.
	NotRetainedInReturnedTopology,
}

/// Exact kernel identity, or an explicit reason that no exact binding exists.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StepTopologyBindingTarget {
	/// Identity of OCCT topology retained by this [`StepImport`]. Exact source
	/// solid targets are present in [`StepImport::solids`]; exact face targets
	/// may instead belong to a display-only body.
	///
	/// This value is stable only while this import's bodies are alive. It identifies
	/// topology, not an assembly occurrence: OCCT locations are deliberately excluded,
	/// so repeated occurrences may share one id.
	Exact {
		kernel_tshape_id: u64,
	},
	Unbound {
		reason: StepTopologyUnboundReason,
	},
}

/// Correspondence from a physical STEP `#id` to imported kernel topology.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StepTopologyBinding {
	pub entity_number: u32,
	pub entity_kind: StepEntityKind,
	pub topology_kind: StepTopologyKind,
	pub target: StepTopologyBindingTarget,
}

/// Placement scope for one source definition in [`StepImport::solids`].
///
/// `definition_to_world` maps the location-stripped `TShape` definition to its
/// occurrence placement. The canonical definition solid does not carry this
/// location; apply the matrix exactly once when lowering the occurrence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepSolidOccurrence {
	/// Index into [`StepImport::solids`] when this occurrence's definition is supported.
	pub definition_index: Option<usize>,
	pub kernel_tshape_id: u64,
	pub definition_to_world: DMat4,
	pub is_identity: bool,
}

impl StepTopologyBinding {
	pub fn kernel_tshape_id(&self) -> Option<u64> {
		match self.target {
			StepTopologyBindingTarget::Exact { kernel_tshape_id } => Some(kernel_tshape_id),
			StepTopologyBindingTarget::Unbound { .. } => None,
		}
	}
}

/// Bounded tessellation presets for a display-only [`StepVisualBody`].
///
/// Arbitrary chord tolerances are deliberately not accepted here: a tiny
/// tolerance on untrusted exchange geometry can request an impractically large
/// mesh. Presets use absolute OCCT kernel millimetres so the standard profile
/// is fidelity-comparable across the Forge STEP corpus. The selected preset
/// also carries a hard output-size check. These meshes remain visual evidence
/// and must not be promoted to collision or mass authority merely because they
/// happen to be closed.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum StepVisualTessellation {
	Preview,
	#[default]
	Standard,
	High,
}

impl StepVisualTessellation {
	fn parameters(self) -> (Tessellation, usize, usize) {
		match self {
			Self::Preview => (Tessellation { deflection_linear: 1.0, deflection_angular: 0.75, relative_linear: false }, 300_000, 100_000),
			Self::Standard => (Tessellation { deflection_linear: 0.1, deflection_angular: 0.25, relative_linear: false }, 1_500_000, 500_000),
			Self::High => (Tessellation { deflection_linear: 0.01, deflection_angular: 0.1, relative_linear: false }, 6_000_000, 2_000_000),
		}
	}
}

/// One opaque face view owned by a display-only STEP body.
///
/// It exposes source-selection identity and effective display color, but no
/// exact `Face` handle. In particular, it cannot be passed to `Solid::sew` or
/// any other geometry-building operation.
#[derive(Clone, Copy)]
pub struct StepVisualFace<'a> {
	_body: &'a StepVisualBody,
	id: u64,
}

impl StepVisualFace<'_> {
	pub fn id(&self) -> u64 {
		self.id
	}

	/// Effective display color: a face color overrides its body's color.
	#[cfg(feature = "color")]
	pub fn color(&self) -> Option<Color> {
		self._body.inner.colormap().get(&self.id).copied().or_else(|| self._body.color())
	}
}

/// A neutral, display-only body recovered during a STEP import.
///
/// The contained OCCT body may have come from a source solid that failed exact
/// retention, a sewn surface model, or another reader recovery path. This type
/// intentionally makes no source-ownership or physical-solid claim. It exposes
/// only identity, opaque face views, read-only color queries, and bounded visual
/// tessellation.
///
/// Physics and CAD operations are absent by construction:
///
/// ```compile_fail
/// # fn physics(body: &cadrum::StepVisualBody) {
/// let _ = body.volume();
/// # }
/// ```
///
/// ```compile_fail
/// # fn convert(body: cadrum::StepVisualBody) {
/// let _: cadrum::Solid = body.into();
/// # }
/// ```
pub struct StepVisualBody {
	inner: Solid,
}

impl StepVisualBody {
	fn new(inner: Solid) -> Self {
		Self { inner }
	}

	/// Import-local OCCT `TShape` identity for display selection and binding lookup.
	pub fn id(&self) -> u64 {
		self.inner.id()
	}

	/// Iterates opaque display faces without exposing CAD-authoring handles.
	pub fn iter_face(&self) -> impl Iterator<Item = StepVisualFace<'_>> + '_ {
		self.inner.iter_face().map(|face| StepVisualFace { _body: self, id: face.id() })
	}

	/// Authored body-level display color, before any face-specific override.
	#[cfg(feature = "color")]
	pub fn color(&self) -> Option<Color> {
		self.inner.colormap().get(&self.id()).copied()
	}

	/// Produces a renderer-oriented mesh under a fixed quality and output budget.
	pub fn mesh(&self, tessellation: StepVisualTessellation) -> Result<Mesh, Error> {
		let (parameters, max_vertices, max_triangles) = tessellation.parameters();
		let mesh = mesh(std::iter::once(&self.inner), parameters)?;
		let vertices = mesh.vertices.len();
		let triangles = mesh.indices.len() / 3;
		if vertices > max_vertices || triangles > max_triangles {
			return Err(Error::VisualMeshBudgetExceeded { vertices, triangles, max_vertices, max_triangles });
		}
		Ok(mesh)
	}
}

/// A STEP import with source-entity topology correspondence.
///
/// Bindings use `TShape` partner identity: orientation and occurrence location are not
/// part of an id. A physical entity number and a kernel id may each occur in multiple
/// rows after healing splits or merges, so callers must group rather than index them.
/// Source face sense is folded into OCCT face orientation and is not recoverable from
/// the id alone. Renderer-oriented normals are available through visual tessellation;
/// exact CAD face queries are exposed only for exact source solids.
/// Ids are process-local, valid only while this import's bodies live, and are invalidated
/// by topology-rebuilding operations. Bindings cover the root STEP file; external-file
/// reader sessions and source entities without a physical `#id` are not represented.
///
/// With the `color` feature, colors remain available through read-only body/face
/// queries. Every body is canonical and location-stripped; [`StepSolidOccurrence`]
/// carries placements for source solid definitions. Bindings come from OCCT's transfer
/// process and sewing history, never vector order or geometric proximity.
pub struct StepImport {
	/// Canonical source solid definitions, deduplicated by exact `TShape` identity.
	pub solids: Vec<Solid>,
	/// Bodies recovered by the reader whose exact source-solid ownership was not
	/// established, but whose retained faces all have exact source bindings.
	///
	/// A body here may originate from a failed source solid or a surface model;
	/// no more specific ownership is claimed. It is display-only by type.
	pub recovered_bodies: Vec<StepVisualBody>,
	/// Bodies whose source topology could not be proven exactly. They are
	/// display-only by type and expose no physical/CAD operations.
	pub unbound_bodies: Vec<StepVisualBody>,
	pub bindings: Vec<StepTopologyBinding>,
	/// Supported source entities skipped because OCCT recorded no physical STEP `#id`.
	pub skipped_identless_entities: u32,
	/// Location scope needed because multiple occurrences may share a `TShape` id.
	pub solid_occurrences: Vec<StepSolidOccurrence>,
}

fn decode_step_bindings(entity_numbers: Vec<u32>, entity_kinds: Vec<u8>, topology_kinds: Vec<u8>, binding_statuses: Vec<u8>, tshape_ids: Vec<u64>) -> Vec<StepTopologyBinding> {
	let len = entity_numbers.len();
	assert_eq!(entity_kinds.len(), len, "Cadrum STEP binding entity-kind length mismatch");
	assert_eq!(topology_kinds.len(), len, "Cadrum STEP binding topology-kind length mismatch");
	assert_eq!(binding_statuses.len(), len, "Cadrum STEP binding status length mismatch");
	assert_eq!(tshape_ids.len(), len, "Cadrum STEP binding TShape-id length mismatch");

	(0..len)
		.map(|index| {
			let entity_kind = match entity_kinds[index] {
				1 => StepEntityKind::ManifoldSolidBrep,
				2 => StepEntityKind::BrepWithVoids,
				3 => StepEntityKind::AdvancedFace,
				4 => StepEntityKind::FacetedBrep,
				5 => StepEntityKind::FacetedBrepWithVoids,
				value => panic!("unknown Cadrum STEP entity-kind encoding {value}"),
			};
			let topology_kind = match topology_kinds[index] {
				1 => StepTopologyKind::Solid,
				2 => StepTopologyKind::Face,
				value => panic!("unknown Cadrum STEP topology-kind encoding {value}"),
			};
			let target = match binding_statuses[index] {
				1 => StepTopologyBindingTarget::Exact { kernel_tshape_id: tshape_ids[index] },
				2 => StepTopologyBindingTarget::Unbound { reason: StepTopologyUnboundReason::NoExpectedTransferResult },
				3 => StepTopologyBindingTarget::Unbound { reason: StepTopologyUnboundReason::NotRetainedInReturnedTopology },
				value => panic!("unknown Cadrum STEP binding-status encoding {value}"),
			};
			StepTopologyBinding { entity_number: entity_numbers[index], entity_kind, topology_kind, target }
		})
		.collect()
}

// ==================== Color trailer ====================
// Appended past the BinTools payload, which BinTools::Read stops at and ignores:
// `[b"CDCL"][u32 count][count x (u32 trailer_ids index, f32 r, f32 g, f32 b)]`, LE.

#[cfg(feature = "color")]
const COLOR_TRAILER_MAGIC: &[u8; 4] = b"CDCL";

/// `tail` is `&buf[consumed..]`, the bytes the BRep parser did not take. Anything that
/// is not our trailer yields an empty map — the geometry is valid either way.
#[cfg(feature = "color")]
fn read_color_trailer(tail: &[u8]) -> std::collections::HashMap<u32, Color> {
	let mut colormap = std::collections::HashMap::new();
	if tail.len() < 8 || &tail[..4] != COLOR_TRAILER_MAGIC {
		return colormap;
	}
	let count = u32::from_le_bytes(tail[4..8].try_into().unwrap()) as usize;
	// `count` comes from the file, and `usize` is 32-bit on wasm32.
	let Some(end) = count.checked_mul(16).and_then(|n| n.checked_add(8)) else {
		return colormap;
	};
	// `<`, not `!=`: the count self-delimits, so bytes appended after us are not an error.
	if tail.len() < end {
		return colormap;
	}
	for e in tail[8..end].chunks_exact(16) {
		let idx = u32::from_le_bytes(e[0..4].try_into().unwrap());
		let r = f32::from_le_bytes(e[4..8].try_into().unwrap());
		let g = f32::from_le_bytes(e[8..12].try_into().unwrap());
		let b = f32::from_le_bytes(e[12..16].try_into().unwrap());
		colormap.insert(idx, Color { r, g, b });
	}
	colormap
}

/// STEP cannot index like this — `try_sew_orphan_faces` shifts every index, so it
/// carries explicit ids instead.
#[cfg(feature = "color")]
fn trailer_ids(shape: &ffi::TopoDS_Shape) -> Vec<u64> {
	// Bound to locals: both are `UniquePtr<CxxVector<..>>` that the iterators borrow.
	let solids = ffi::decompose_into_solids(shape);
	let faces = ffi::shape_faces(shape);
	solids.iter().map(ffi::shape_tshape_id).chain(faces.iter().map(ffi::face_tshape_id)).collect()
}

#[cfg(feature = "color")]
fn write_color_trailer<W: Write>(compound: &CompoundShape, writer: &mut W) -> Result<(), Error> {
	let id_to_index: std::collections::HashMap<u64, u32> = trailer_ids(compound.inner()).into_iter().enumerate().map(|(i, id)| (id, i as u32)).collect();
	// `CompoundShape::decompose` gives every solid a clone of the merged colormap, so
	// a solid carries its siblings' keys too; those have no index and drop out here.
	let mut entries: Vec<(u32, f32, f32, f32)> = compound.colormap().iter().filter_map(|(id, rgb)| id_to_index.get(id).map(|&idx| (idx, rgb.r, rgb.g, rgb.b))).collect();
	if entries.is_empty() {
		return Ok(());
	}
	entries.sort_by_key(|e| e.0);

	let mut out = Vec::with_capacity(8 + entries.len() * 16);
	out.extend_from_slice(COLOR_TRAILER_MAGIC);
	out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
	for (idx, r, g, b) in &entries {
		out.extend_from_slice(&idx.to_le_bytes());
		out.extend_from_slice(&r.to_le_bytes());
		out.extend_from_slice(&g.to_le_bytes());
		out.extend_from_slice(&b.to_le_bytes());
	}
	writer.write_all(&out).map_err(|_| Error::BrepWriteFailed)
}

// ==================== Reader / writer / mesh helpers ====================
//
// Each function is invoked by the matching `SolidStruct` method in
// `super::solid::Solid`. Kept module-private (`pub(super)`) so the public
// surface lives entirely on `Solid`.

pub(super) fn read_step<R: Read>(reader: &mut R) -> Result<Vec<Solid>, Error> {
	#[cfg(feature = "color")]
	{
		let mut rust_reader = RustReader::from_ref(reader);
		let mut ids: Vec<u64> = Default::default();
		let mut rgb: Vec<f32> = Default::default();
		let inner = ffi::read_step_color_stream(&mut rust_reader, &mut ids, &mut rgb);
		if inner.is_null() {
			return Err(Error::StepReadFailed);
		}
		let colormap: std::collections::HashMap<u64, Color> = ids.into_iter().zip(rgb.chunks_exact(3)).map(|(id, c)| (id, Color { r: c[0], g: c[1], b: c[2] })).collect();
		Ok(CompoundShape::from_raw(inner, colormap, Default::default()).decompose())
	}
	#[cfg(not(feature = "color"))]
	{
		let mut rust_reader = RustReader::from_ref(reader);
		let inner = ffi::read_step_stream(&mut rust_reader);
		if inner.is_null() {
			return Err(Error::StepReadFailed);
		}
		Ok(CompoundShape::from_raw(inner, Default::default()).decompose())
	}
}

pub(crate) fn read_step_with_bindings<R: Read>(reader: &mut R) -> Result<StepImport, Error> {
	let mut entity_numbers = Vec::new();
	let mut entity_kinds = Vec::new();
	let mut topology_kinds = Vec::new();
	let mut binding_statuses = Vec::new();
	let mut tshape_ids = Vec::new();
	let mut occurrence_tshape_ids = Vec::new();
	let mut occurrence_matrices = Vec::new();
	let mut skipped_identless_entities = 0;

	#[cfg(feature = "color")]
	let (inner, colormap) = {
		let mut rust_reader = RustReader::from_ref(reader);
		let mut ids = Vec::new();
		let mut rgb = Vec::new();
		let inner = ffi::read_step_bound_color_stream(&mut rust_reader, &mut ids, &mut rgb, &mut entity_numbers, &mut entity_kinds, &mut topology_kinds, &mut binding_statuses, &mut tshape_ids, &mut occurrence_tshape_ids, &mut occurrence_matrices, &mut skipped_identless_entities);
		if inner.is_null() {
			return Err(Error::StepReadFailed);
		}
		let colormap = ids.into_iter().zip(rgb.chunks_exact(3)).map(|(id, c)| (id, Color { r: c[0], g: c[1], b: c[2] })).collect();
		(inner, colormap)
	};

	#[cfg(not(feature = "color"))]
	let inner = {
		let mut rust_reader = RustReader::from_ref(reader);
		let inner = ffi::read_step_bound_stream(&mut rust_reader, &mut entity_numbers, &mut entity_kinds, &mut topology_kinds, &mut binding_statuses, &mut tshape_ids, &mut occurrence_tshape_ids, &mut occurrence_matrices, &mut skipped_identless_entities);
		if inner.is_null() {
			return Err(Error::StepReadFailed);
		}
		inner
	};

	let bodies = CompoundShape::from_raw(
		inner,
		#[cfg(feature = "color")]
		colormap,
		Default::default(),
	)
	.decompose();
	let bindings = decode_step_bindings(entity_numbers, entity_kinds, topology_kinds, binding_statuses, tshape_ids);
	let solid_binding_ids: std::collections::HashSet<u64> = bindings.iter().filter(|binding| binding.topology_kind == StepTopologyKind::Solid).filter_map(StepTopologyBinding::kernel_tshape_id).collect();
	let face_binding_ids: std::collections::HashSet<u64> = bindings.iter().filter(|binding| binding.topology_kind == StepTopologyKind::Face).filter_map(StepTopologyBinding::kernel_tshape_id).collect();
	let mut solids = Vec::new();
	let mut recovered_bodies = Vec::new();
	let mut unbound_bodies = Vec::new();
	for body in bodies {
		if solid_binding_ids.contains(&body.id()) {
			solids.push(body);
		} else {
			let face_ids: Vec<u64> = body.iter_face().map(|face| face.id()).collect();
			if !face_ids.is_empty() && face_ids.iter().all(|id| face_binding_ids.contains(id)) {
				recovered_bodies.push(StepVisualBody::new(body));
			} else {
				unbound_bodies.push(StepVisualBody::new(body));
			}
		}
	}
	assert_eq!(occurrence_matrices.len(), occurrence_tshape_ids.len() * 16, "Cadrum STEP occurrence-matrix length mismatch");
	let definition_indices: std::collections::HashMap<u64, usize> = solids.iter().enumerate().map(|(index, solid)| (ffi::shape_tshape_id(solid.inner()), index)).collect();
	let solid_occurrences = occurrence_tshape_ids
		.into_iter()
		.zip(occurrence_matrices.chunks_exact(16))
		.map(|(kernel_tshape_id, values)| {
			let values: [f64; 16] = values.try_into().expect("chunks_exact(16) must yield 16 values");
			let definition_to_world = DMat4::from_cols_array(&values);
			StepSolidOccurrence { definition_index: definition_indices.get(&kernel_tshape_id).copied(), kernel_tshape_id, definition_to_world, is_identity: definition_to_world == DMat4::IDENTITY }
		})
		.collect();
	Ok(StepImport { solids, recovered_bodies, unbound_bodies, bindings, skipped_identless_entities, solid_occurrences })
}

pub(super) fn read_brep<R: Read>(reader: &mut R) -> Result<Vec<Solid>, Error> {
	// Buffered whole: `BinTools::Read` seeks backwards to resolve shared sub-shape
	// references, so it cannot run off a sequential stream.
	let mut buf = Vec::new();
	reader.read_to_end(&mut buf).map_err(|_| Error::BrepReadFailed)?;

	// Payload length — where a trailer would begin. Unwritten, and unread, on null.
	let mut consumed = 0usize;
	let inner = ffi::read_brep_stream(&buf, &mut consumed);
	if inner.is_null() {
		return Err(Error::BrepReadFailed);
	}

	#[cfg(feature = "color")]
	{
		let ids = trailer_ids(&inner);
		let colormap = read_color_trailer(buf.get(consumed..).unwrap_or_default()).into_iter().filter_map(|(idx, color)| ids.get(idx as usize).map(|&id| (id, color))).collect();
		Ok(CompoundShape::from_raw(inner, colormap, Default::default()).decompose())
	}
	#[cfg(not(feature = "color"))]
	{
		Ok(CompoundShape::from_raw(inner, Default::default()).decompose())
	}
}

/// Write solids to a STEP stream.
///
/// With the `color` feature enabled, face colors are automatically embedded
/// in the STEP file (XDE / AP214 styled items).
pub(super) fn write_step<'a, W: Write>(solids: impl IntoIterator<Item = &'a Solid>, writer: &mut W) -> Result<(), Error> {
	let compound = CompoundShape::new(solids);
	#[cfg(feature = "color")]
	{
		let colormap = compound.colormap();
		let mut ids: Vec<u64> = Vec::with_capacity(colormap.len());
		let mut rgb: Vec<f32> = Vec::with_capacity(colormap.len() * 3);
		for (&id, c) in colormap {
			ids.push(id);
			rgb.extend_from_slice(&[c.r, c.g, c.b]);
		}
		let mut rust_writer = RustWriter::from_ref(writer);
		if ffi::write_step_color_stream(compound.inner(), &ids, &rgb, &mut rust_writer) {
			Ok(())
		} else {
			Err(Error::StepWriteFailed)
		}
	}
	#[cfg(not(feature = "color"))]
	{
		let mut rust_writer = RustWriter::from_ref(writer);
		if ffi::write_step_stream(compound.inner(), &mut rust_writer) {
			Ok(())
		} else {
			Err(Error::StepWriteFailed)
		}
	}
}

pub(super) fn write_brep<'a, W: Write>(solids: impl IntoIterator<Item = &'a Solid>, writer: &mut W) -> Result<(), Error> {
	let compound = CompoundShape::new(solids);
	{
		// Scoped: the streambuf flushes on drop, so the payload lands before the trailer.
		let mut rust_writer = RustWriter::from_ref(writer);
		if !ffi::write_brep_stream(compound.inner(), &mut rust_writer) {
			return Err(Error::BrepWriteFailed);
		}
	}
	#[cfg(feature = "color")]
	write_color_trailer(&compound, writer)?;
	Ok(())
}

pub(super) fn mesh<'a>(solids: impl IntoIterator<Item = &'a Solid>, options: crate::traits::Tessellation) -> Result<crate::common::mesh::Mesh, Error> {
	use crate::common::mesh::Mesh;
	use glam::DVec3;

	#[cfg(feature = "color")]
	let solids: Vec<&Solid> = solids.into_iter().collect();
	// `Mesh` has only a face level, so a solid-level colour is expanded onto its faces
	// here. STEP and the BRep trailer keep the distinction; the renderers cannot.
	#[cfg(feature = "color")]
	let face_colors = {
		let mut map = std::collections::HashMap::new();
		for s in solids.iter().copied() {
			if let Some(&c) = s.colormap().get(&s.id()) {
				for f in ffi::shape_faces(s.inner()).iter() {
					map.insert(ffi::face_tshape_id(f), c);
				}
			}
			// Face colours are the more specific style and win over the solid's.
			map.extend(s.colormap().iter().map(|(&k, &v)| (k, v)));
		}
		map
	};

	let compound = CompoundShape::new(solids);
	let data = ffi::mesh_shape(compound.inner(), options.deflection_linear, options.deflection_angular, options.relative_linear);
	if !data.success {
		return Err(Error::TriangulationFailed);
	}
	let vertex_count = data.vertices.len() / 3;
	let vertices: Vec<DVec3> = (0..vertex_count).map(|i| DVec3::new(data.vertices[i * 3], data.vertices[i * 3 + 1], data.vertices[i * 3 + 2])).collect();
	let normals: Vec<DVec3> = (0..vertex_count).map(|i| DVec3::new(data.normals[i * 3], data.normals[i * 3 + 1], data.normals[i * 3 + 2])).collect();
	let indices: Vec<usize> = data.indices.iter().map(|&i| i as usize).collect();
	let face_ids = data.face_tshape_ids;

	// Topological edge polylines, NaN-separated. Reuses the existing edge
	// discretizer (GCPnts_TangentialDeflection). `relative_linear` applies to
	// surface triangulation only; edges use `deflection_linear` as an absolute
	// chord here.
	let mut edges: Vec<DVec3> = Vec::new();
	for e in ffi::shape_edges(compound.inner()).iter() {
		let segs = ffi::edge_approximation_segments(e, options.deflection_linear, options.deflection_angular, options.relative_linear);
		if segs.len() < 6 {
			continue; // fewer than 2 points — nothing to draw
		}
		if !edges.is_empty() {
			edges.push(DVec3::NAN);
		}
		for c in segs.chunks_exact(3) {
			edges.push(DVec3::new(c[0], c[1], c[2]));
		}
	}

	#[cfg(feature = "color")]
	let colormap = {
		let mut map = std::collections::HashMap::new();
		for &fid in &face_ids {
			if let Some(&color) = face_colors.get(&fid) {
				map.insert(fid, color);
			}
		}
		map
	};

	Ok(Mesh {
		vertices,
		normals,
		indices,
		face_ids,
		#[cfg(feature = "color")]
		colormap,
		edges,
	})
}

#[cfg(test)]
mod step_binding_tests {
	use super::*;

	#[test]
	fn decodes_every_exact_solid_entity_kind_without_collapsing_faceted_voids() {
		let bindings = decode_step_bindings(vec![10, 11, 12, 13], vec![1, 2, 4, 5], vec![1, 1, 1, 1], vec![1, 1, 1, 1], vec![100, 101, 102, 103]);
		assert_eq!(bindings.iter().map(|binding| binding.entity_kind).collect::<Vec<_>>(), [StepEntityKind::ManifoldSolidBrep, StepEntityKind::BrepWithVoids, StepEntityKind::FacetedBrep, StepEntityKind::FacetedBrepWithVoids,]);
	}

	#[test]
	fn visual_step_profiles_are_explicit_absolute_kernel_millimetres() {
		let (preview, _, _) = StepVisualTessellation::Preview.parameters();
		let (standard, _, _) = StepVisualTessellation::Standard.parameters();
		let (high, _, _) = StepVisualTessellation::High.parameters();
		assert_eq!((preview.deflection_linear, preview.deflection_angular, preview.relative_linear), (1.0, 0.75, false));
		assert_eq!((standard.deflection_linear, standard.deflection_angular, standard.relative_linear), (0.1, 0.25, false));
		assert_eq!((high.deflection_linear, high.deflection_angular, high.relative_linear), (0.01, 0.1, false));
	}
}
