
import can
bus = can.Bus(interface='udp_multicast', channel='239.0.0.1', fd=True)
msg = can.Message(arbitration_id=0x100, data=b'\x01\x02\x03\x04', is_fd=True)
bus.send(msg)
print('TX done')
bus.shutdown()
