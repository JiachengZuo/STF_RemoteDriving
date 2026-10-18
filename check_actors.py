import carla
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(5.0)
world = client.get_world()
vehicles = world.get_actors().filter('vehicle.*')
print(f"Vehicles in world: {len(vehicles)}")
for v in vehicles:
    loc = v.get_location()
    role = v.attributes.get('role_name', '?')
    print(f"  id={v.id} role={role} type={v.type_id} at ({loc.x:.1f},{loc.y:.1f},{loc.z:.1f})")
