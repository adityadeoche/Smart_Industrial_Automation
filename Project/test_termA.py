import can, time
bus = can.Bus(interface='udp_multicast', channel='239.0.0.1', fd=True)
for msg in bus:
    print('RX:', msg)