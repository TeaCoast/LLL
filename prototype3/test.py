import math

hello = 0.1115
print(hello + 0.0000000000000000)

mantissa, exponent = math.frexp(hello)

print("Mantissa", int(mantissa*2**53))
print("Exponent", exponent-53)