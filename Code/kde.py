def kde(z_list, z_query, h, kernel):
    sum = 0
    for z in z_list:
        sum += kernel.get_kernel_value((z_query-z)/h)
    return sum/(len(z_list)*h)


