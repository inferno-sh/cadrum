//! Chamfer integration tests with analytic cube checks and released Articraft
//! selected-edge fixtures.

use cadrum::{DVec3, EdgeSelector, Error, Solid};

const EPS: f64 = 1e-6;
type TopologySignature = (Vec<[[u64; 3]; 4]>, Vec<([u64; 3], u64)>);

#[test]
fn test_chamfer_cube_reduces_volume_and_area() {
	let a = 10.0_f64;
	let d = 1.0_f64;
	let cube = Solid::cube(DVec3::ZERO, DVec3::splat(a));
	let original_volume = cube.volume();
	let original_area = cube.area();
	let edges: Vec<_> = cube.iter_edge().collect();
	let beveled = cube.chamfer_edges(d, edges).expect("chamfer should succeed");

	assert!(beveled.volume() < original_volume, "chamfer must reduce volume: {} vs {}", beveled.volume(), original_volume);
	assert!(beveled.area() < original_area, "chamfer must reduce area: {} vs {}", beveled.area(), original_area);
}

#[test]
fn test_chamfer_cube_matches_analytical_volume() {
	let a = 10.0_f64;
	let d = 1.0_f64;
	let cube = Solid::cube(DVec3::ZERO, DVec3::splat(a));
	let edges: Vec<_> = cube.iter_edge().collect();
	let beveled = cube.chamfer_edges(d, edges).expect("chamfer cube");

	// Inclusion-exclusion:
	//   12 edge-wedges (d²/2 × a)       = 6 a d²
	//   − 24 corner pair-intersections  = 8 d³
	//   + 8 corner triple-intersections = 2 d³
	//   = 6 d² (a − d)
	let expected = a.powi(3) - 6.0 * d * d * (a - d);
	let rel_err = (beveled.volume() - expected).abs() / expected;
	assert!(rel_err < 1e-3, "chamfered cube volume {} vs analytical {} (rel err {})", beveled.volume(), expected, rel_err);
}

#[test]
fn test_chamfer_empty_edges_is_noop() {
	let cube = Solid::cube(DVec3::ZERO, DVec3::splat(5.0));
	let original_volume = cube.volume();
	let unchanged = cube.chamfer_edges(0.5, std::iter::empty::<&cadrum::Edge>()).expect("empty chamfer is a no-op, not an error");
	assert!((unchanged.volume() - original_volume).abs() < EPS, "no-op chamfer should preserve volume exactly, got {}", unchanged.volume());
}

#[test]
fn test_chamfer_distance_too_large_returns_err() {
	let cube = Solid::cube(DVec3::ZERO, DVec3::splat(2.0));
	let edges: Vec<_> = cube.iter_edge().collect();
	// d = 5 > a/2 = 1 → geometrically impossible; OCCT reports not-done.
	let err = cube.chamfer_edges(5.0, edges).err().expect("oversized distance must fail");
	assert!(matches!(err, Error::ChamferFailed), "expected ChamferFailed, got {:?}", err);
}

fn selected_chamfer_metrics(size: DVec3, distance: f64, axis: DVec3) -> (f64, f64) {
	let (section_a, section_b, length) = if axis.x.abs() > 0.5 {
		(size.y, size.z, size.x)
	} else if axis.y.abs() > 0.5 {
		(size.x, size.z, size.y)
	} else {
		(size.x, size.y, size.z)
	};
	let section_area = section_a * section_b - 2.0 * distance.powi(2);
	let section_perimeter = 2.0 * (section_a + section_b) - 4.0 * (2.0 - 2.0_f64.sqrt()) * distance;
	(section_area * length, 2.0 * section_area + section_perimeter * length)
}

fn point_bits(point: DVec3) -> [u64; 3] {
	[point.x.to_bits(), point.y.to_bits(), point.z.to_bits()]
}

fn ordered_topology_signature(solid: &Solid) -> TopologySignature {
	let edges = solid.iter_edge().map(|edge| [point_bits(edge.start_point()), point_bits(edge.end_point()), point_bits(edge.start_tangent()), point_bits(edge.end_tangent())]).collect();
	let faces = solid.iter_face().map(|face| (point_bits(face.center()), face.area().to_bits())).collect();
	(edges, faces)
}

fn assert_selected_box_chamfer(size: DVec3, distance: f64, axis: DVec3) -> Solid {
	let solid = Solid::cube(DVec3::ZERO, size).chamfer_selected_edges(distance, EdgeSelector::parallel_to(axis)).expect("selected chamfer");
	let (expected_volume, expected_area) = selected_chamfer_metrics(size, distance, axis);

	assert!((solid.volume() - expected_volume).abs() < 1.0e-12, "volume={} expected={expected_volume}", solid.volume());
	assert!((solid.area() - expected_area).abs() < 1.0e-11, "area={} expected={expected_area}", solid.area());
	assert!((solid.center() - size * 0.5).length() < 1.0e-12);
	assert_eq!(solid.iter_face().count(), 10);
	assert_eq!(solid.iter_edge().count(), 24);

	let [minimum, maximum] = solid.bounding_box();
	assert!((minimum - DVec3::ZERO).abs().max_element() < 1.0e-6, "minimum={minimum:?}");
	assert!((maximum - size).abs().max_element() < 1.0e-6, "maximum={maximum:?}");
	solid
}

#[test]
fn parallel_axis_selector_chamfers_all_box_axes_analytically() {
	let size = DVec3::new(3.0, 4.0, 5.0);
	for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
		assert_selected_box_chamfer(size, 0.3, axis);
	}
}

#[test]
fn parallel_axis_selector_matches_released_barrier_gate_chamfer() {
	let chamfered = assert_selected_box_chamfer(DVec3::new(1.05, 0.95, 0.32), 0.035, DVec3::Z);
	assert!((chamfered.volume() - 0.3184159999999999).abs() < 1.0e-14);
}

#[test]
fn parallel_axis_chamfer_preserves_topology_order() {
	let cube = Solid::cube(DVec3::ZERO, DVec3::new(3.0, 4.0, 5.0));
	let positive = cube.chamfer_selected_edges(0.3, EdgeSelector::parallel_to(DVec3::Z)).expect("positive axis");
	let repeated = cube.chamfer_selected_edges(0.3, EdgeSelector::parallel_to(DVec3::Z)).expect("repeated axis");
	let negative = cube.chamfer_selected_edges(0.3, EdgeSelector::parallel_to(DVec3::NEG_Z)).expect("negative axis");
	assert_eq!(ordered_topology_signature(&positive), ordered_topology_signature(&repeated));
	assert_eq!(ordered_topology_signature(&positive), ordered_topology_signature(&negative));
}

#[test]
fn parallel_axis_chamfer_rejects_invalid_inputs_and_empty_matches() {
	let cube = Solid::cube(DVec3::ZERO, DVec3::splat(2.0));
	for distance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
		assert!(matches!(cube.chamfer_selected_edges(distance, EdgeSelector::parallel_to(DVec3::Z)), Err(Error::ChamferFailed)));
	}

	let invalid_selector = EdgeSelector::ParallelToAxis { axis: DVec3::Z, angular_tolerance: f64::NAN };
	assert!(matches!(cube.chamfer_selected_edges(0.2, invalid_selector), Err(Error::EdgeSelectionFailed(_))));
	let diagonal = DVec3::new(1.0, 1.0, 0.0).normalize();
	assert!(matches!(cube.chamfer_selected_edges(0.2, EdgeSelector::parallel_to(diagonal)), Err(Error::EdgeSelectionFailed(_))));
}

#[test]
fn extreme_selector_matches_analytic_cadquery_box_chamfers() {
	let size = DVec3::new(3.0, 4.0, 5.0);
	let cube = Solid::cube(DVec3::ZERO, size);
	for (axis, face_a, face_b, expected_volume) in [(DVec3::X, size.y, size.z, 59.226), (DVec3::Y, size.x, size.z, 59.316), (DVec3::Z, size.x, size.y, 59.406)] {
		let maximum = cube.chamfer_selected_edges(0.3, EdgeSelector::at_maximum(axis)).expect("maximum chamfer");
		let minimum = cube.chamfer_selected_edges(0.3, EdgeSelector::at_minimum(axis)).expect("minimum chamfer");
		let analytic = size.x * size.y * size.z - 0.3_f64.powi(2) * (face_a + face_b) + 4.0 / 3.0 * 0.3_f64.powi(3);
		assert!((maximum.volume() - analytic).abs() < 1.0e-12);
		assert!((maximum.volume() - expected_volume).abs() < 1.0e-12);
		assert!((minimum.volume() - maximum.volume()).abs() < 1.0e-12);
		for solid in [&maximum, &minimum] {
			assert_eq!(solid.iter_face().count(), 10);
			assert_eq!(solid.iter_edge().count(), 20);
			let [box_minimum, box_maximum] = solid.bounding_box();
			assert!((box_minimum - DVec3::ZERO).abs().max_element() < 1.0e-6);
			assert!((box_maximum - size).abs().max_element() < 1.0e-6);
		}
	}
}

#[test]
fn extreme_chamfer_preserves_topology_order() {
	let cube = Solid::cube(DVec3::ZERO, DVec3::new(3.0, 4.0, 5.0));
	let first = cube.chamfer_selected_edges(0.3, EdgeSelector::at_maximum(DVec3::Z)).expect("first >Z chamfer");
	let second = cube.chamfer_selected_edges(0.3, EdgeSelector::at_maximum(DVec3::Z)).expect("second >Z chamfer");
	assert_eq!(ordered_topology_signature(&first), ordered_topology_signature(&second));
}
