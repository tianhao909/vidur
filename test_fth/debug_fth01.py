import debugpy

# # Listen for debugger connections on a specified port
# debugpy.listen(5678)
# # Wait for a debugger client to attach
# debugpy.wait_for_client()

def calculate_sum(a, b):
    result = a + b  # Set a breakpoint here
    return result

# Call the function and print the result
print(calculate_sum(10, 20))