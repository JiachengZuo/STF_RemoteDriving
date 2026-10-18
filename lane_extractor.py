#!/usr/bin/env python3
# Remote Driving Framework - Lane Extractor
# Extracts lane waypoints from CARLA map for BEV rendering.
# Sent once at startup, then cached by relay/operator.

import carla
import math


def extract_lanes(world, sample_distance=5.0, max_distance=500.0, center_only=False):
    """Extract lane centerlines and boundaries from CARLA waypoints.

    Args:
        world: CARLA world object
        sample_distance: spacing between waypoints (meters)
        max_distance: maximum distance from spawn to extract (meters)
        center_only: if True, only extract centerlines (not lane boundaries)

    Returns:
        dict with:
            'centerlines': list of list of (x, y) — lane center polylines
            'boundaries': list of list of (x, y) — lane boundary polylines
            'lane_types': list of lane types (driving, sidewalk, etc.)
    """
    carla_map = world.get_map()
    topology = carla_map.get_topology()

    # Get all spawn points to find the area of interest
    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        return {'centerlines': [], 'boundaries': [], 'lane_types': []}

    # Compute bounding box of spawn points
    xs = [sp.location.x for sp in spawn_points]
    ys = [sp.location.y for sp in spawn_points]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    radius = max(
        max(abs(x - cx) for x in xs),
        max(abs(y - cy) for y in ys)
    ) * 2  # generous margin

    centerlines = []
    boundaries = []
    lane_types = []
    visited_segments = set()

    for segment in topology:
        wp_start = segment[0]
        wp_end = segment[1]

        # Check if segment is near spawn area
        if (abs(wp_start.transform.location.x - cx) > radius or
                abs(wp_start.transform.location.y - cy) > radius):
            continue

        seg_key = (round(wp_start.transform.location.x, 1),
                   round(wp_start.transform.location.y, 1),
                   round(wp_end.transform.location.x, 1),
                   round(wp_end.transform.location.y, 1))
        if seg_key in visited_segments:
            continue
        visited_segments.add(seg_key)

        # Walk along the lane
        pts = []
        max_steps = int(max_distance / sample_distance)
        current = wp_start
        for _ in range(max_steps):
            loc = current.transform.location
            pts.append((round(loc.x, 2), round(loc.y, 2)))
            next_wps = current.next(sample_distance)
            if not next_wps:
                break
            current = next_wps[0]

        if len(pts) >= 2:
            lane_type = wp_start.lane_type
            type_str = str(lane_type)
            if type_str in ('Driving', 'Shoulder', 'Sidewalk', 'Parking'):
                centerlines.append(pts)
                lane_types.append(type_str)

            # Extract lane boundaries
            if not center_only and type_str == 'Driving':
                left_bounds = []
                right_bounds = []
                current = wp_start
                for _ in range(max_steps):
                    loc = current.transform.location
                    yaw = math.radians(current.transform.rotation.yaw)
                    lw = current.lane_width / 2.0
                    # Left boundary
                    left_bounds.append((
                        round(loc.x + lw * math.cos(yaw + math.pi / 2), 2),
                        round(loc.y + lw * math.sin(yaw + math.pi / 2), 2)))
                    # Right boundary
                    right_bounds.append((
                        round(loc.x + lw * math.cos(yaw - math.pi / 2), 2),
                        round(loc.y + lw * math.sin(yaw - math.pi / 2), 2)))
                    next_wps = current.next(sample_distance)
                    if not next_wps:
                        break
                    current = next_wps[0]

                if len(left_bounds) >= 2:
                    boundaries.append(left_bounds)
                    boundaries.append(right_bounds)

    return {
        'centerlines': centerlines,
        'boundaries': boundaries,
        'lane_types': lane_types,
        'bbox': [round(cx - radius, 1), round(cy - radius, 1),
                 round(cx + radius, 1), round(cy + radius, 1)],
        'num_lanes_summary': f"{len(centerlines)} centerlines, {len(boundaries)} boundaries"
    }
