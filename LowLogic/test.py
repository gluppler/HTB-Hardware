import csv

values = None
binres = ''

def output(inputs):
    # The logic
    result = (int(inputs[0]) and int(inputs[1])) or (int(inputs[2]) and int(inputs[3]))
    return result

with open('input.csv', mode='r') as csvfile:
    values = csv.reader(csvfile)
    for value in values:
        if value[0] != 'in0':
            print(f"{value}: {output(value)}")
            binres += str(output(value))

# Convert binary to string
print(int(binres, 2).to_bytes((199) // 8, 'big').decode())
