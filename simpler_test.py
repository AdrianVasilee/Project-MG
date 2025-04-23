from layers.leaky_ReLU import LeakyReLU

myReLU = LeakyReLU(0.1)

values = range(-10, 10)

for i in values:
    print(f"For {i}: {myReLU.forwards(i)}")