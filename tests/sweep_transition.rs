use cadrum::{DVec3, Edge, ProfileOrient, Solid, SweepTransition};

fn rectangle_in_yz(width: f64, height: f64) -> Vec<Edge> {
	let half_width = width * 0.5;
	let half_height = height * 0.5;
	Edge::polygon(&[DVec3::new(0.0, -half_width, -half_height), DVec3::new(0.0, half_width, -half_height), DVec3::new(0.0, half_width, half_height), DVec3::new(0.0, -half_width, half_height)]).expect("rectangle profile")
}

fn c0_right_angle_spine() -> [Edge; 2] {
	[Edge::line(DVec3::ZERO, DVec3::X * 10.0).expect("first spine edge"), Edge::line(DVec3::X * 10.0, DVec3::new(10.0, 10.0, 0.0)).expect("second spine edge")]
}

#[test]
fn sweep_keeps_transformed_as_the_default_transition() {
	let profile = rectangle_in_yz(2.0, 1.0);
	let spine = c0_right_angle_spine();
	let implicit = Solid::sweep(&profile, &spine, ProfileOrient::Up(DVec3::Z)).expect("implicit transformed sweep");
	let explicit = Solid::sweep_with_transition(&profile, &spine, ProfileOrient::Up(DVec3::Z), SweepTransition::Transformed).expect("explicit transformed sweep");

	assert!((implicit.volume() - explicit.volume()).abs() < 1.0e-9);
	assert!((implicit.area() - explicit.area()).abs() < 1.0e-9);
	assert_eq!(implicit.bounding_box(), explicit.bounding_box());
}

#[test]
fn c0_corner_transition_modes_produce_distinct_solids() {
	let profile = rectangle_in_yz(2.0, 1.0);
	let spine = c0_right_angle_spine();
	let transformed = Solid::sweep_with_transition(&profile, &spine, ProfileOrient::Up(DVec3::Z), SweepTransition::Transformed).expect("transformed sweep");
	let round = Solid::sweep_with_transition(&profile, &spine, ProfileOrient::Up(DVec3::Z), SweepTransition::Round).expect("round sweep");
	let right = Solid::sweep_with_transition(&profile, &spine, ProfileOrient::Up(DVec3::Z), SweepTransition::Right).expect("right sweep");

	assert!(transformed.volume() < round.volume());
	assert!(round.volume() < right.volume());
	assert!((right.volume() - round.volume()).abs() > 0.1);
	assert!((round.volume() - transformed.volume()).abs() > 20.0);
	assert_ne!(transformed.bounding_box(), round.bounding_box());
	assert_eq!(SweepTransition::default(), SweepTransition::Transformed);
}
