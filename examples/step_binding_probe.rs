use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;

use cadrum::{Solid, StepTopologyBindingTarget, StepVisualTessellation};

fn main() {
	let path = env::args().nth(1).expect("usage: step_binding_probe FILE.step [ENTITY] [PROFILE]");
	let selected_entity = env::args().nth(2).map(|value| value.parse::<u32>().expect("ENTITY must be an unsigned integer"));
	let profiles: Vec<_> = match env::args().nth(3).as_deref() {
		None | Some("standard") => vec![StepVisualTessellation::Standard],
		Some("preview") => vec![StepVisualTessellation::Preview],
		Some("high") => vec![StepVisualTessellation::High],
		Some("all") => vec![StepVisualTessellation::Preview, StepVisualTessellation::Standard, StepVisualTessellation::High],
		Some(_) => panic!("PROFILE must be preview, standard, high, or all"),
	};
	let bytes = fs::read(&path).expect("read STEP file");
	let import = Solid::read_step_with_bindings(&mut bytes.as_slice()).expect("import STEP bindings");
	let sources_by_target = import.bindings.iter().fold(BTreeMap::<u64, Vec<u32>>::new(), |mut result, binding| {
		if let StepTopologyBindingTarget::Exact { kernel_tshape_id } = binding.target {
			result.entry(kernel_tshape_id).or_default().push(binding.entity_number);
		}
		result
	});

	println!("exact_solids={} recovered_bodies={} unbound_bodies={} bindings={} identless={}", import.solids.len(), import.recovered_bodies.len(), import.unbound_bodies.len(), import.bindings.len(), import.skipped_identless_entities);
	for (body_index, body) in import.recovered_bodies.iter().enumerate() {
		for &profile in &profiles {
			let mesh = body.mesh(profile).expect("mesh recovered body");
			let meshed = mesh.face_ids.iter().copied().collect::<BTreeSet<_>>();
			let mesh_area_by_face = mesh.indices.chunks_exact(3).zip(&mesh.face_ids).fold(BTreeMap::<u64, f64>::new(), |mut areas, (triangle, &face_id)| {
				let a = mesh.vertices[triangle[0]];
				let b = mesh.vertices[triangle[1]];
				let c = mesh.vertices[triangle[2]];
				*areas.entry(face_id).or_default() += 0.5 * (b - a).cross(c - a).length();
				areas
			});
			println!("body={body_index} profile={profile:?} vertices={} triangles={} meshed_faces={}", mesh.vertices.len(), mesh.indices.len() / 3, meshed.len());
			let unmeshed: Vec<_> = body.iter_face().filter(|face| !meshed.contains(&face.id())).collect();
			for face in &unmeshed {
				println!("body={body_index} profile={profile:?} unmeshed_tshape={} area_mm2={} source_entities={:?}", face.id(), face.visual_area_mm2(), sources_by_target.get(&face.id()).cloned().unwrap_or_default());
			}
			if let Some(entity) = selected_entity {
				for face in body.iter_face().filter(|face| sources_by_target.get(&face.id()).is_some_and(|sources| sources.contains(&entity))) {
					println!("body={body_index} profile={profile:?} source_entity={entity} exact_area_mm2={} triangle_area_mm2={}", face.visual_area_mm2(), mesh_area_by_face.get(&face.id()).copied().unwrap_or_default());
				}
			}
			assert!(unmeshed.is_empty(), "every retained visual face must emit triangles");
		}
	}
}
