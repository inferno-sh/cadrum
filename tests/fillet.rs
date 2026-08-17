//! Fillet integration tests with analytic cube checks and released Articraft
//! selected-edge fixtures.

use cadrum::{AxisExtreme, DVec3, EdgeSelector, Error, Solid};
use std::f64::consts::PI;

const EPS: f64 = 1e-6;
type TopologySignature = (Vec<[[u64; 3]; 4]>, Vec<([u64; 3], u64)>);

#[test]
fn test_fillet_cube_reduces_volume_and_area() {
	let a = 10.0_f64;
	let r = 1.0_f64;
	let cube = Solid::cube(DVec3::ZERO, DVec3::splat(a));
	let original_volume = cube.volume();
	let original_area = cube.area();
	let edges: Vec<_> = cube.iter_edge().collect();
	let rounded = cube.fillet_edges(r, edges).expect("fillet should succeed");

	assert!(rounded.volume() < original_volume, "fillet must reduce volume: {} vs {}", rounded.volume(), original_volume);
	assert!(rounded.area() < original_area, "fillet must reduce area: {} vs {}", rounded.area(), original_area);
}

#[test]
fn test_fillet_cube_matches_analytical_volume() {
	let a = 10.0_f64;
	let r = 1.0_f64;
	let cube = Solid::cube(DVec3::ZERO, DVec3::splat(a));
	let edges: Vec<_> = cube.iter_edge().collect();
	let rounded = cube.fillet_edges(r, edges).expect("fillet cube");

	let removed_edges = 12.0 * (r * r - PI * r * r / 4.0) * (a - 2.0 * r);
	let removed_corners = 8.0 * (r.powi(3) - PI * r.powi(3) / 6.0);
	let expected = a.powi(3) - removed_edges - removed_corners;
	let rel_err = (rounded.volume() - expected).abs() / expected;
	assert!(rel_err < 1e-3, "rounded cube volume {} vs analytical {} (rel err {})", rounded.volume(), expected, rel_err);
}

#[test]
fn test_fillet_empty_edges_is_noop() {
	let cube = Solid::cube(DVec3::ZERO, DVec3::splat(5.0));
	let original_volume = cube.volume();
	let unchanged = cube.fillet_edges(0.5, std::iter::empty::<&cadrum::Edge>()).expect("empty fillet is a no-op, not an error");
	assert!((unchanged.volume() - original_volume).abs() < EPS, "no-op fillet should preserve volume exactly, got {}", unchanged.volume());
}

#[test]
fn test_fillet_radius_too_large_returns_err() {
	let cube = Solid::cube(DVec3::ZERO, DVec3::splat(2.0));
	let edges: Vec<_> = cube.iter_edge().collect();
	// r = 5 > a/2 = 1 → geometrically impossible; OCCT reports not-done.
	let err = cube.fillet_edges(5.0, edges).err().expect("oversized radius must fail");
	assert!(matches!(err, Error::FilletFailed), "expected FilletFailed, got {:?}", err);
}

fn rounded_prism_metrics(size: DVec3, radius: f64) -> (f64, f64) {
	let rounded_section_area = size.x * size.y - (4.0 - PI) * radius.powi(2);
	let rounded_section_perimeter = 2.0 * (size.x + size.y) - (8.0 - 2.0 * PI) * radius;
	(rounded_section_area * size.z, 2.0 * rounded_section_area + rounded_section_perimeter * size.z)
}

fn point_bits(point: DVec3) -> [u64; 3] {
	[point.x.to_bits(), point.y.to_bits(), point.z.to_bits()]
}

fn ordered_topology_signature(solid: &Solid) -> TopologySignature {
	let edges = solid.iter_edge().map(|edge| [point_bits(edge.start_point()), point_bits(edge.end_point()), point_bits(edge.start_tangent()), point_bits(edge.end_tangent())]).collect();
	let faces = solid.iter_face().map(|face| (point_bits(face.center()), face.area().to_bits())).collect();
	(edges, faces)
}

fn assert_parallel_z_box_fillet(size: DVec3, radius: f64) -> Solid {
	let solid = Solid::cube(DVec3::ZERO, size).fillet_selected_edges(radius, EdgeSelector::parallel_to(DVec3::Z)).expect("parallel-Z fillet");
	let (expected_volume, expected_area) = rounded_prism_metrics(size, radius);

	assert!((solid.volume() - expected_volume).abs() < 1.0e-14, "volume={} expected={expected_volume}", solid.volume());
	assert!((solid.area() - expected_area).abs() < 1.0e-13, "area={} expected={expected_area}", solid.area());
	assert!((solid.center() - size * 0.5).length() < 1.0e-13);
	assert_eq!(solid.iter_face().count(), 10);
	assert_eq!(solid.iter_edge().count(), 24);

	let [minimum, maximum] = solid.bounding_box();
	assert!((minimum - DVec3::ZERO).abs().max_element() < 1.0e-6, "minimum={minimum:?}");
	assert!((maximum - size).abs().max_element() < 1.0e-6, "maximum={maximum:?}");
	solid
}

#[test]
fn parallel_axis_selector_matches_analytic_rounded_prism() {
	assert_parallel_z_box_fillet(DVec3::new(3.0, 4.0, 5.0), 0.3);
}

#[test]
fn parallel_axis_selector_matches_released_faucet_tower_and_pad() {
	let tower = assert_parallel_z_box_fillet(DVec3::new(0.046, 0.050, 0.046), 0.0045);
	let pad = assert_parallel_z_box_fillet(DVec3::new(0.010, 0.020, 0.022), 0.0016);
	assert!((tower.volume() - 0.00010500039355681892).abs() < 1.0e-16);
	assert!((pad.volume() - 4.351654498250177e-6).abs() < 1.0e-17);
}

#[test]
fn parallel_axis_selector_preserves_topology_order() {
	let cube = Solid::cube(DVec3::ZERO, DVec3::new(3.0, 4.0, 5.0));
	let positive = cube.fillet_selected_edges(0.3, EdgeSelector::parallel_to(DVec3::Z)).expect("positive axis");
	let repeated = cube.fillet_selected_edges(0.3, EdgeSelector::parallel_to(DVec3::Z)).expect("repeated axis");
	let negative = cube.fillet_selected_edges(0.3, EdgeSelector::parallel_to(DVec3::NEG_Z)).expect("negative axis");
	assert_eq!(ordered_topology_signature(&positive), ordered_topology_signature(&repeated));
	assert_eq!(ordered_topology_signature(&positive), ordered_topology_signature(&negative));
}

#[test]
fn parallel_axis_selector_rejects_invalid_inputs_and_empty_matches() {
	let cube = Solid::cube(DVec3::ZERO, DVec3::splat(2.0));
	for radius in [0.0, -1.0, f64::NAN, f64::INFINITY] {
		assert!(matches!(cube.fillet_selected_edges(radius, EdgeSelector::parallel_to(DVec3::Z)), Err(Error::FilletFailed)));
	}

	for selector in [EdgeSelector::parallel_to(DVec3::ZERO), EdgeSelector::parallel_to(DVec3::Z * 2.0), EdgeSelector::parallel_to(DVec3::new(f64::NAN, 0.0, 1.0)), EdgeSelector::ParallelToAxis { axis: DVec3::Z, angular_tolerance: 0.0 }, EdgeSelector::ParallelToAxis { axis: DVec3::Z, angular_tolerance: f64::INFINITY }, EdgeSelector::ParallelToAxis { axis: DVec3::Z, angular_tolerance: std::f64::consts::FRAC_PI_2 }] {
		assert!(matches!(cube.fillet_selected_edges(0.2, selector), Err(Error::EdgeSelectionFailed(_))));
	}

	let diagonal = DVec3::new(1.0, 1.0, 0.0).normalize();
	let no_match = cube.fillet_selected_edges(0.2, EdgeSelector::parallel_to(diagonal));
	assert!(matches!(no_match, Err(Error::EdgeSelectionFailed(message)) if message.contains("matched no edges")));
}

#[test]
fn extreme_selector_matches_cadquery_top_and_bottom_box_fillets() {
	let cube = Solid::cube(DVec3::ZERO, DVec3::new(3.0, 4.0, 5.0));
	let top = cube.fillet_selected_edges(0.3, EdgeSelector::at_maximum(DVec3::Z)).expect(">Z fillet");
	let bottom = cube.fillet_selected_edges(0.3, EdgeSelector::at_minimum(DVec3::Z)).expect("<Z fillet");

	assert!((top.volume() - 59.73995570185034).abs() < 1.0e-7);
	assert!((top.area() - 92.14637124991249).abs() < 1.0e-7);
	assert!((bottom.volume() - top.volume()).abs() < 1.0e-7);
	assert!((top.center().z + bottom.center().z - 5.0).abs() < 1.0e-7);
	for solid in [&top, &bottom] {
		assert_eq!(solid.iter_face().count(), 10);
		assert_eq!(solid.iter_edge().count(), 20);
		let [minimum, maximum] = solid.bounding_box();
		assert!((minimum - DVec3::ZERO).abs().max_element() < 1.0e-6);
		assert!((maximum - DVec3::new(3.0, 4.0, 5.0)).abs().max_element() < 1.0e-6);
	}
}

#[test]
fn extreme_selector_is_deterministic_and_validates_model_space_tolerance() {
	let cube = Solid::cube(DVec3::ZERO, DVec3::new(3.0, 4.0, 5.0));
	let selector = EdgeSelector::AtExtreme { axis: DVec3::Z, extreme: AxisExtreme::Maximum, linear_tolerance: 1.0e-6 };
	let first = cube.fillet_selected_edges(0.3, selector).expect("first >Z fillet");
	let second = cube.fillet_selected_edges(0.3, selector).expect("second >Z fillet");
	assert_eq!(ordered_topology_signature(&first), ordered_topology_signature(&second));

	for invalid in [EdgeSelector::AtExtreme { axis: DVec3::Z * 2.0, extreme: AxisExtreme::Maximum, linear_tolerance: 1.0e-6 }, EdgeSelector::AtExtreme { axis: DVec3::Z, extreme: AxisExtreme::Maximum, linear_tolerance: 0.0 }, EdgeSelector::AtExtreme { axis: DVec3::Z, extreme: AxisExtreme::Maximum, linear_tolerance: f64::NAN }] {
		assert!(matches!(cube.fillet_selected_edges(0.3, invalid), Err(Error::EdgeSelectionFailed(_))));
	}
}
