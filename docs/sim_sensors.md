# Virtual wrist and tactile signals

`ContactSensorSuite` provides read-only measurements from a fresh MuJoCo state.
It adds no sites, joints, constraints or collision geometry and changes no
native gains, inertia or contact masks. The existing scene can keep MuJoCo's
ordinary sensor flag disabled: the suite copies data and computes post-constraint
inverse dynamics on its own scratch object.

```python
from abstract_stack.sim_sensors import ContactSensorSuite

suite = ContactSensorSuite.mabel(env.model)
metadata = suite.metadata()
sample = suite.sample(env.data)  # after native stepping/forward
```

| Channel | Shape | Meaning |
| --- | --- | --- |
| `wrist_wrench` | 2 × 6 | Native parent-interaction force/torque, in left/right wrist frames |
| `contact_wrench` | 2 × 6 | Contact-only net wrench **on the distal robot subtree**, in the same frames |
| `tactile` | N × 3 | Coarse force **on each configured collision geom**, in its geom frame |
| `timestamp_s` | scalar | Native simulation sample time |
| Availability masks | 2 and N | Present configured channels; zero force with a true mask means no contact |

Wrench order is `[Fx, Fy, Fz, Tx, Ty, Tz]`, with N then Nm. The wrist origin is
the native wrist joint anchor; its axes follow the joint's child body. On MABEL,
these are the two palms. Parent-interaction readings match MuJoCo's built-in
force/torque sensor convention and include gravity/inertial loading. They are
not interchangeable with contact-only force on the robot or force on a carton.
The implementation transports native COM-based moments to the wrist origin
before rotating the result. See the primary
[force/torque sensor reference](https://mujoco.readthedocs.io/en/3.17.0/XMLreference.html#sensor-force).

Contact-only measurements include all contacts crossing a wrist subtree
boundary. Contacts internal to that hand cancel at its wrist but still appear
in the per-geom tactile proxies. Tangential contact forces are retained; native
contact couples contribute to wrist torque. Tactile force-only cells do not
represent those couples as independent torque measurements.

The native MABEL fixture exposes 56 collision-geom force cells. They are an
explicit rigid-geometry approximation, **not calibrated skin taxels**, surface
pressure, deformation or optical tactile images. The suite currently has no
noise, bias, saturation, quantization or delay model. Provenance makes these
limitations explicit; hardware parity has not been established.

The caller owns sample cadence and must supply data from the admitted model
after its kinematics/contacts are fresh. The suite does not advance physics,
repair stale contacts or silently replace missing observations. Its arrays are
copies; channel order/frame metadata is available through `metadata()`.
Student/teacher datasets retain this provenance and timestamp/availability
information through [the LfD pipeline](learning_from_demonstration.md).

Nine focused tests cover exact native F/T agreement in rotated frames with and
without contact, contact-force/moment signs, force conservation, internal-hand
contact cancellation, sensor-disabled scenes, configuration rejection and both
native MABEL wrists. Live integration state and derived arrays remain unchanged
by sampling. These are sensor-contract tests, not manipulation-success evidence.
