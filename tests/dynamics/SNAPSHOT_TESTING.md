# Snapshot Testing for Orbit Dynamics

This directory contains snapshot tests for orbit dynamics calculations. These tests capture the current behavior of the orbit propagators and sun dynamics to prevent unintended changes during refactoring.

## What are Snapshot Tests?

Snapshot tests capture the output of your functions and compare them against stored "snapshots" on subsequent test runs. If the output changes, the test fails, alerting you to potential breaking changes in your refactoring.

## Test Coverage

The snapshot tests cover:

### Circular Orbit Propagator (`CircularSatelliteOrbitPropagator`)

- Equatorial orbits (0° inclination)
- Polar orbits (90° inclination)
- Inclined orbits (e.g., ISS-like at 51.6°)
- Full orbital period propagation
- Geostationary orbit altitude
- LEO constellation orbits (e.g., Starlink-like)
- Time propagation from epoch

### Elliptical Orbit Propagator (`EllipticalSatelliteOrbitPropagator`)

- Circular case (zero eccentricity)
- Moderately eccentric orbits (e=0.2)
- Highly eccentric orbits (Molniya-like, e=0.72)
- ISS-like orbit parameters
- Different reference frames (TEME, J2000, GCRS)
- Geostationary Transfer Orbit (GTO)

### Sun Dynamics (`get_sun_direction_eci`)

- Single time point calculations
- Daily variations (24-hour period)
- Seasonal variations
- Yearly propagation
- Equinoxes and solstices
- Intraday variations

## Running the Tests

### Run all snapshot tests

```bash
pytest tests/dynamics/test_orbit_snapshot.py -v
```

### Run specific tests by pattern

```bash
# Run all circular orbit tests
pytest tests/dynamics/test_orbit_snapshot.py -k "circular" -v

# Run all elliptical orbit tests
pytest tests/dynamics/test_orbit_snapshot.py -k "elliptical" -v

# Run all sun direction tests
pytest tests/dynamics/test_orbit_snapshot.py -k "sun_direction" -v
```

### Run a specific test

```bash
pytest tests/dynamics/test_orbit_snapshot.py::test_circular_equatorial_orbit_snapshot -v
```

## When Refactoring

### 1. Run Tests Before Refactoring

```bash
pytest tests/dynamics/test_orbit_snapshot.py -v
```

All tests should pass, confirming the current baseline.

### 2. Perform Your Refactoring

Make changes to the orbit dynamics code in `src/cosmica/dynamics/`.

### 3. Run Tests After Refactoring

```bash
pytest tests/dynamics/test_orbit_snapshot.py -v
```

### 4. Interpret Results

#### ✅ All tests pass

Your refactoring preserved the existing behavior. Great!

#### ❌ Tests fail with snapshot mismatches

This means your refactoring changed the output. You have two options:

**Option A: Fix the code** - If the change was unintentional, debug and fix your changes.

**Option B: Update snapshots** - If the change was intentional and correct:

```bash
# Review the differences carefully first
pytest tests/dynamics/test_orbit_snapshot.py -v

# If changes are correct, update the snapshots
pytest tests/dynamics/test_orbit_snapshot.py --snapshot-update
```

⚠️ **IMPORTANT**: Only update snapshots if you're confident the new behavior is correct!

## Understanding Snapshot Files

Snapshots are stored in `tests/dynamics/__snapshots__/test_orbit_snapshot.ambr`.

This file contains human-readable representations of:

- Position vectors (in ECI frame, meters)
- Velocity vectors (in ECI frame, m/s)
- Sun direction vectors (unit vectors in ECI frame)

Example snapshot format:

```ambr
# name: test_circular_equatorial_orbit_snapshot
  SatelliteOrbitState(
    position_eci=
      [[7000000.0, 0.0, 0.0],
       [6985362.6, 452447.6, 0.0],
       ...]
    velocity_eci=
      [[0.0, 7546.0, 0.0],
       [-487.7, 7530.2, 0.0],
       ...]
  )
```

## Best Practices

1. **Review snapshot diffs carefully** - Don't blindly update snapshots. Understand why the output changed.

2. **Keep snapshots in version control** - Commit the `__snapshots__` directory along with your code changes.

3. **Run tests frequently** - Run snapshot tests after each significant change to catch issues early.

4. **Use with unit tests** - Snapshot tests complement (not replace) traditional unit tests. Keep both!

5. **Document intentional changes** - When updating snapshots due to intentional changes, document why in your commit message.

## Troubleshooting

### Tests pass locally but fail in CI

- Ensure all dependencies are synchronized
- Check for platform-specific numerical differences (though these tests use high precision to minimize this)

### Snapshot file is too large

- The current snapshots are comprehensive. If they grow too large, consider:
  - Splitting into multiple test files
  - Reducing the number of time steps in some tests

### Numerical precision issues

- Arrays are rounded to 7 decimal places before serialization
- This prevents platform-specific floating-point precision differences between local and CI environments
- The rounding ensures snapshots are identical across different platforms while maintaining sufficient precision for orbit calculations

## Additional Resources

- [Syrupy Documentation](https://github.com/tophat/syrupy)
- [Snapshot Testing Best Practices](https://kentcdodds.com/blog/effective-snapshot-testing)
