# Function to update weights incrementally
def weight_update(instances, prev_weight, z_query, z, kernel, h=1.0):
    kernel_value = kernel.get_kernel_value((z_query-z)/h)
    new_weight = prev_weight + (1/instances+1)*((kernel_value/h) - prev_weight)
    return new_weight