use cadrum::{DVec3, Solid, StepEntityKind, StepTopologyKind, StepVisualBody, StepVisualTessellation};
use std::collections::HashSet;

fn colored_box() -> Vec<u8> {
	std::fs::read("steps/colored_box.step").expect("colored STEP fixture should exist")
}

#[test]
fn binds_physical_step_ids_to_exact_definition_topology() {
	let import = Solid::read_step_with_bindings(&mut colored_box().as_slice()).expect("bound STEP read should succeed");
	assert_eq!(import.solids.len(), 1);
	assert!(import.recovered_bodies.is_empty());
	assert!(import.unbound_bodies.is_empty());
	assert_eq!(import.skipped_identless_entities, 0);
	assert_eq!(import.solid_occurrences.len(), 1);
	assert_eq!(import.solid_occurrences[0].definition_index, Some(0));
	assert!(import.solid_occurrences[0].is_identity);

	let topology_ids: HashSet<u64> = import.solids.iter().flat_map(|solid| std::iter::once(solid.id()).chain(solid.iter_face().map(|face| face.id()))).collect();
	assert_eq!(import.bindings.len(), 12, "one MANIFOLD_SOLID_BREP and eleven ADVANCED_FACE entities");
	assert!(import.bindings.iter().all(|binding| binding.kernel_tshape_id().is_some_and(|id| topology_ids.contains(&id))));
	let advanced_face_numbers: HashSet<u32> = import.bindings.iter().filter(|binding| binding.entity_kind == StepEntityKind::AdvancedFace).map(|binding| binding.entity_number).collect();
	assert_eq!(advanced_face_numbers, (197..=207).collect(), "physical STEP ids must not be replaced with dense model ranks");

	let solid = import.bindings.iter().find(|binding| binding.entity_number == 13).expect("physical STEP #13 binding");
	assert_eq!(solid.entity_kind, StepEntityKind::ManifoldSolidBrep);
	assert_eq!(solid.topology_kind, StepTopologyKind::Solid);
	assert_eq!(solid.kernel_tshape_id(), Some(import.solids[0].id()));

	let face = import.bindings.iter().find(|binding| binding.entity_number == 197).expect("physical STEP #197 binding");
	assert_eq!(face.entity_kind, StepEntityKind::AdvancedFace);
	assert_eq!(face.topology_kind, StepTopologyKind::Face);
}

#[test]
fn faceted_brep_retains_its_exact_source_kind() {
	let source = String::from_utf8(colored_box()).expect("fixture is ASCII STEP");
	let faceted = source.replace("MANIFOLD_SOLID_BREP", "FACETED_BREP");
	assert!(faceted.contains("FACETED_BREP") && !faceted.contains("MANIFOLD_SOLID_BREP"));

	let import = Solid::read_step_with_bindings(&mut faceted.as_bytes()).expect("faceted STEP read should succeed");
	assert_eq!(import.solids.len(), 1);
	let binding = import.bindings.iter().find(|binding| binding.entity_number == 13).expect("physical FACETED_BREP #13 binding");
	assert_eq!(binding.entity_kind, StepEntityKind::FacetedBrep);
	assert_eq!(binding.topology_kind, StepTopologyKind::Solid);
	assert_eq!(binding.kernel_tshape_id(), Some(import.solids[0].id()));
}

#[test]
fn brep_with_voids_is_not_collapsed_into_a_plain_manifold_brep() {
	let outer = Solid::cube(DVec3::ZERO, DVec3::splat(4.0));
	let inner = Solid::cube(DVec3::splat(1.0), DVec3::splat(3.0));
	let hollow: Solid = (&outer - &inner).build().expect("nested subtraction should produce one voided solid");
	let mut bytes = Vec::new();
	Solid::write_step([&hollow], &mut bytes).expect("voided solid STEP write should succeed");
	assert!(String::from_utf8_lossy(&bytes).contains("BREP_WITH_VOIDS"));

	let import = Solid::read_step_with_bindings(&mut bytes.as_slice()).expect("voided STEP read should succeed");
	assert!(import.bindings.iter().any(|binding| binding.entity_kind == StepEntityKind::BrepWithVoids && binding.topology_kind == StepTopologyKind::Solid && binding.kernel_tshape_id().is_some()));
	assert!(!import.bindings.iter().any(|binding| matches!(binding.entity_kind, StepEntityKind::FacetedBrep | StepEntityKind::FacetedBrepWithVoids)));
}

#[test]
fn reader_recovery_never_becomes_a_bound_solid_definition() {
	let source = String::from_utf8(colored_box()).expect("fixture is ASCII STEP");
	let surface = source.replace("ADVANCED_BREP_SHAPE_REPRESENTATION", "MANIFOLD_SURFACE_SHAPE_REPRESENTATION").replace("#13=MANIFOLD_SOLID_BREP('\\X2\\30DC30C730A3\\X0\\1',#208);", "#13=SHELL_BASED_SURFACE_MODEL('\\X2\\30DC30C730A3\\X0\\1',(#208));").replace("#208=CLOSED_SHELL(", "#208=OPEN_SHELL(");
	assert!(!surface.contains("MANIFOLD_SOLID_BREP"));
	assert!(surface.contains("SHELL_BASED_SURFACE_MODEL") && surface.contains("OPEN_SHELL"));

	let legacy = Solid::read_step(&mut surface.as_bytes()).expect("legacy read should recover the closed face set");
	assert_eq!(legacy.len(), 1, "legacy compatibility path synthesizes one solid");

	let import = Solid::read_step_with_bindings(&mut surface.as_bytes()).expect("bound STEP read should succeed");
	assert!(import.solids.is_empty(), "surface entities are not source-declared solid definitions");
	assert!(import.solid_occurrences.is_empty(), "surface recovery is not a source solid occurrence");
	assert_eq!(import.recovered_bodies.len(), 1, "reader recovery remains available through a neutral display-only type");
	assert!(import.unbound_bodies.is_empty(), "sewing history should prove every recovered face");
	assert_eq!(import.skipped_identless_entities, 0);
	assert!(!import.bindings.iter().any(|binding| binding.topology_kind == StepTopologyKind::Solid));
	let recovered: &StepVisualBody = &import.recovered_bodies[0];
	let recovered_face_ids: HashSet<u64> = recovered.iter_face().map(|face| face.id()).collect();
	assert!(recovered.iter_face().all(|face| face.visual_area_mm2().is_finite() && face.visual_area_mm2() >= 0.0));
	assert!(import.bindings.iter().all(|binding| binding.kernel_tshape_id().is_some_and(|id| recovered_face_ids.contains(&id))));
	#[cfg(feature = "color")]
	assert!(recovered.iter_face().any(|face| face.color().is_some()), "visual face color remains queryable without exposing a CAD Face");
	let mesh = recovered.mesh(StepVisualTessellation::Preview).expect("display-only body should tessellate within the preview budget");
	assert!(!mesh.indices.is_empty());
}

#[test]
fn sewing_recovery_preserves_advanced_face_provenance() {
	let data = std::fs::read("steps/multicolor_solvespace.step").expect("SolveSpace STEP fixture should exist");
	let import = Solid::read_step_with_bindings(&mut data.as_slice()).expect("bound STEP read should succeed");
	assert!(import.unbound_bodies.is_empty());
	assert_eq!(import.skipped_identless_entities, 0);

	let body_face_ids: HashSet<u64> = import.solids.iter().flat_map(|body| body.iter_face().map(|face| face.id())).chain(import.recovered_bodies.iter().flat_map(|body| body.iter_face().map(|face| face.id()))).collect();
	let face_bindings: Vec<_> = import.bindings.iter().filter(|binding| binding.entity_kind == StepEntityKind::AdvancedFace).collect();
	assert_eq!(face_bindings.len(), 15);
	assert!(face_bindings.iter().all(|binding| binding.kernel_tshape_id().is_some_and(|id| body_face_ids.contains(&id))));
}
