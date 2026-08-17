//! Articraft-10K stool `83fc…355e2`, `model.py:76-78`: a 0.01 XZ circle at X=0.18 swept around a 0.18 XY circle.
//! CadQuery's omitted arguments mean `makeSolid=true`, `isFrenet=false`, and `transition="right"`.

use cadrum::{DVec3, Edge, Mesh, ProfileOrient, Solid, SweepTransition, Tessellation};
use std::collections::HashSet;

const PATH_RADIUS: f64 = 0.18;
const PROFILE_RADIUS: f64 = 0.01;
const MESH_DEFLECTION: f64 = 0.0005;

fn articraft_stool_footrest_ring() -> Solid {
	let spine = Edge::circle(PATH_RADIUS, DVec3::Z).expect("closed XY circle spine");
	let profile = Edge::circle(PROFILE_RADIUS, DVec3::NEG_Y).expect("XZ circle profile").translate(DVec3::X * PATH_RADIUS);

	assert!(spine.is_closed());
	assert!(profile.is_closed());
	assert!((spine.start_point() - DVec3::X * PATH_RADIUS).length() < 1.0e-15);
	assert!((spine.start_tangent() - DVec3::Y).length() < 1.0e-15);

	Solid::sweep_with_transition(&[profile], &[spine], ProfileOrient::Corrected, SweepTransition::Right).expect("closed corrected-Frenet solid sweep")
}

fn tessellation() -> Tessellation {
	Tessellation { deflection_linear: MESH_DEFLECTION, deflection_angular: 0.1, relative_linear: false }
}

fn point_bits(points: &[DVec3]) -> Vec<[u64; 3]> {
	points.iter().map(|point| [point.x.to_bits(), point.y.to_bits(), point.z.to_bits()]).collect()
}

fn assert_identical_mesh(actual: &Mesh, expected: &Mesh) {
	assert_eq!(point_bits(&actual.vertices), point_bits(&expected.vertices));
	assert_eq!(point_bits(&actual.normals), point_bits(&expected.normals));
	assert_eq!(actual.indices, expected.indices);
	assert_eq!(actual.face_ids, expected.face_ids);
	// `edges` contains deliberate NaN separators, so compare bit patterns.
	assert_eq!(point_bits(&actual.edges), point_bits(&expected.edges));
}

#[test]
fn closed_circle_spine_matches_articraft_geometry_and_topology() {
	let ring = articraft_stool_footrest_ring();
	let expected_volume = 2.0 * std::f64::consts::PI.powi(2) * PATH_RADIUS * PROFILE_RADIUS.powi(2);
	let expected_area = 4.0 * std::f64::consts::PI.powi(2) * PATH_RADIUS * PROFILE_RADIUS;

	assert!((ring.volume() - expected_volume).abs() < 1.0e-15, "volume={} expected={expected_volume}", ring.volume());
	assert!((ring.area() - expected_area).abs() < 1.0e-14, "area={} expected={expected_area}", ring.area());
	assert!(ring.center().length() < 1.0e-14, "center={:?}", ring.center());

	// One periodic face and two seam edges prove a closed solid without start/end
	// cap faces.
	assert_eq!(ring.iter_face().count(), 1);
	assert_eq!(ring.iter_edge().count(), 2);
	assert_eq!(ring.iter_face().next().expect("periodic face").iter_edge().count(), 2);

	// OCCT's general Bnd_Box query is conservative for this periodic swept face;
	// it must enclose the analytic torus box but need not be tight.
	let [minimum, maximum] = ring.bounding_box();
	let outer_radius = PATH_RADIUS + PROFILE_RADIUS;
	assert!(minimum.x <= -outer_radius && minimum.y <= -outer_radius && minimum.z <= -PROFILE_RADIUS);
	assert!(maximum.x >= outer_radius && maximum.y >= outer_radius && maximum.z >= PROFILE_RADIUS);

	let mesh = Solid::mesh([&ring], tessellation()).expect("closed sweep must mesh");
	assert!(!mesh.vertices.is_empty());
	assert_eq!(mesh.indices.len() % 3, 0);
	assert_eq!(mesh.face_ids.len(), mesh.indices.len() / 3);
	assert_eq!(mesh.face_ids.iter().copied().collect::<HashSet<_>>().len(), 1);

	let [mesh_minimum, mesh_maximum] = mesh.vertices.iter().fold([DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)], |[minimum, maximum], point| [minimum.min(*point), maximum.max(*point)]);
	let expected_minimum = DVec3::new(-outer_radius, -outer_radius, -PROFILE_RADIUS);
	let expected_maximum = -expected_minimum;
	assert!((mesh_minimum - expected_minimum).abs().max_element() <= MESH_DEFLECTION);
	assert!((mesh_maximum - expected_maximum).abs().max_element() <= MESH_DEFLECTION);
}

#[test]
fn closed_circle_spine_mesh_is_bitwise_repeatable() {
	let ring = articraft_stool_footrest_ring();
	let first = Solid::mesh([&ring], tessellation()).expect("first mesh");
	let second = Solid::mesh([&ring], tessellation()).expect("repeated mesh");
	assert_identical_mesh(&second, &first);

	// Rebuilding the exact B-rep gives fresh topology identities, but its mesh
	// ordering and floating-point geometry remain deterministic.
	let rebuilt = articraft_stool_footrest_ring();
	let rebuilt_mesh = Solid::mesh([&rebuilt], tessellation()).expect("rebuilt mesh");
	assert_eq!(point_bits(&rebuilt_mesh.vertices), point_bits(&first.vertices));
	assert_eq!(point_bits(&rebuilt_mesh.normals), point_bits(&first.normals));
	assert_eq!(rebuilt_mesh.indices, first.indices);
	assert_eq!(point_bits(&rebuilt_mesh.edges), point_bits(&first.edges));
}
