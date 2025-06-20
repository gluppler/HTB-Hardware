import numpy as np
import socket
import base64
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from lascar.container import AcquisitionFromGetters 
from lascar import CpaEngine, Session
from lascar.tools.aes import sbox


# --- Network Configuration ---
# Set HOST and PORT for the target server.
# These values are reverted to default local testing values as requested.
HOST = '0.0.0.0'
PORT = 1337

# --- Global Data for CPA Attack ---
# HW (Hamming Weight) array: Pre-calculates the Hamming weight for all byte values (0-255).
# This is crucial for the power model in a CPA attack, as it estimates the power consumption
# based on the number of '1' bits.
# Converted to a NumPy array with dtype=np.uint8 for Numba compatibility and efficiency.
HW = np.array([bin(n).count("1") for n in range(256)], dtype=np.uint8)

# --- Base64 Decoding Function ---
def b64_decode_trace(leakage):
    """
    Decodes a base64-encoded byte string into a NumPy array representing a power trace.
    
    Args:
        leakage (bytes): The base64-encoded power trace received from the server.
        
    Returns:
        numpy.ndarray: The decoded power trace as a NumPy array.
        
    Raises:
        ValueError: If base64 decoding fails, NumPy conversion fails,
                    or the resulting power trace is empty.
    """
    try:
        # Decode the base64 string to raw byte data
        byte_data = base64.b64decode(leakage)
        # Convert the raw byte data into a NumPy array of numerical values
        trace_array = np.frombuffer(byte_data)
        
        # Critical check: Ensure the decoded trace is not empty.
        # An empty trace can cause 'ValueError: could not broadcast input array from shape (0,)'
        # errors later in lascar's processing.
        if trace_array.size == 0:
            raise ValueError("Decoded power trace is empty.")
            
        return trace_array
    except Exception as e:
        # Catch any exceptions during decoding/conversion and re-raise as ValueError
        raise ValueError(f"Base64 decode or numpy conversion failed: {e}")

# --- Server Interaction Function ---
def interact_with_server(option: bytes, data: bytes) -> bytes:
    """
    Manages communication with the remote target server via a TCP socket.
    
    This function handles establishing a connection, sending an option (e.g., '1' for trace, '2' for key),
    sending the corresponding data (plaintext or key), and receiving the server's response.
    
    Args:
        option (bytes): The byte string representing the server option ('1' or '2').
        data (bytes): The data to send based on the chosen option (plaintext or hex-encoded key).
        
    Returns:
        bytes: The raw response data received from the server (e.g., base64-encoded trace, flag).
               Returns None if any network or communication error occurs.
    """
    try:
        # Create a new socket connection. `with` statement ensures proper closing.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Set a timeout for all socket operations (connect, send, recv).
            # This prevents the script from hanging indefinitely if the server is unresponsive.
            s.settimeout(5.0) 

            # Connect to the specified host and port
            s.connect((HOST, PORT))

            # Receive initial banner/welcome message from the server.
            # Content is ignored, as we only need to progress the communication state.
            s.recv(1024) 

            # Send the chosen option ('1' or '2') to the server.
            # No newline character is added, as the server expects raw byte commands.
            s.sendall(option)

            # Receive the prompt from the server after sending the option.
            # Content is ignored, as we only need to progress the communication state.
            s.recv(1024) 

            # Send the actual data (plaintext or key) to the server.
            s.sendall(data)

            # Receive the full response from the server.
            # This loop continues receiving chunks of data until the server closes the connection
            # or no more data is available (indicated by an empty chunk).
            resp_data = b''
            try:
                while True:
                    temp_data = s.recv(8096) # Receive data in larger chunks for efficiency
                    if not temp_data:
                        break # Server closed connection or sent no more data
                    resp_data += temp_data
                    # Small delay to yield control and allow server to send all data,
                    # can sometimes help with network stability over flaky connections.
                    time.sleep(0.001) 
            except socket.timeout:
                # If a timeout occurs during data reception, consider it a failure.
                return None 

            s.close() # Explicitly close the socket.
            return resp_data
    except socket.timeout as e:
        # Handle timeouts specifically for connection establishment or initial sends.
        return None 
    except Exception as e:
        # Catch any other general network-related exceptions.
        return None 

# --- Plaintext Generation Function ---
def random_ascii_plaintext():
    """
    Generates a random 16-byte plaintext.
    
    This is critical for CPA attacks, as varying plaintexts produce varying
    intermediate values and thus varying power traces, which are necessary
    for statistical analysis to differentiate signal from noise.
    
    Returns:
        bytes: A 16-byte randomly generated plaintext.
    """
    return bytes([random.randint(0, 255) for _ in range(16)])

# --- Single Trace Collection Function ---
def collect_single_trace(i, retries=3):
    """
    Attempts to collect a single power trace and its corresponding plaintext from the server.
    
    Includes a retry mechanism to handle transient network issues or malformed responses.
    
    Args:
        i (int): A unique identifier for the trace collection attempt (for logging).
        retries (int): The number of times to retry if collection fails.
        
    Returns:
        tuple: A tuple (plaintext, power_trace_numpy_array) on success.
               Returns an error string on failure after all retries are exhausted.
    """
    for attempt in range(retries):
        pt = random_ascii_plaintext() # Generate a new random plaintext for each attempt
        raw = interact_with_server(b'1', pt) # Request a trace for this plaintext
        
        if raw is None:
            # If interact_with_server returns None, it indicates a network/connection issue.
            time.sleep(0.1) # Wait briefly before retrying for network stability
            continue # Try again if retries remain
        
        try:
            # Attempt to decode the raw response into a NumPy array trace.
            # b64_decode_trace now includes checks for empty arrays.
            trace = b64_decode_trace(raw) 
            return (pt, trace) # Return successful plaintext and trace
        except ValueError as e:
            # Catch specific ValueErrors (e.g., bad base64, empty trace)
            time.sleep(0.1) # Wait briefly before retry
            continue # Try again if retries remain
        except Exception as e:
            # Catch any other unexpected errors during processing
            time.sleep(0.1)
            continue

    # If all retries fail, return a descriptive error message
    return f"Failed after {retries} attempts: check server or network. (Trace ID {i})"

# --- Parallel Trace Collection Function ---
def collect_traces_parallel(n=1000, workers=20, max_overall_attempts_factor=5):
    """
    Collects a specified number of power traces ('n') in parallel using a ThreadPoolExecutor.
    
    This function is designed to be robust against individual trace collection failures
    by continuously submitting new tasks until 'n' successful traces are acquired
    or a global limit on total attempts is reached.
    
    Args:
        n (int): The target number of successful traces to collect.
        workers (int): The maximum number of concurrent threads for collection.
        max_overall_attempts_factor (int): Multiplier for 'n' to set the hard limit
                                           on total collection attempts.
    
    Returns:
        tuple: A tuple (list_of_plaintexts, list_of_traces) containing all successfully collected data.
    """
    print(f"[*] Collecting {n} traces with {workers} threads...")
    traces = []
    pts = []
    current_trace_id = 0 # Unique ID for each trace collection request (useful for debugging)
    successful_traces_count = 0
    total_attempts_made = 0
    # Calculate the maximum total attempts allowed to prevent infinite loops
    max_total_attempts = n * max_overall_attempts_factor 

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {} # Dictionary to hold active futures: {future_object: unique_trace_id}

        # Initial population of the worker pool: submit enough tasks to fill the workers
        while len(futures) < workers and current_trace_id < max_total_attempts:
            future = executor.submit(collect_single_trace, current_trace_id)
            futures[future] = current_trace_id
            current_trace_id += 1
            total_attempts_made += 1

        # Main loop to manage futures and collect traces
        while successful_traces_count < n and futures:
            # Use as_completed to get futures as they complete, allowing dynamic submission
            for future in as_completed(futures.keys()):
                original_trace_id = futures.pop(future) # Remove the completed future from active list
                result = future.result() # Get the result of the completed task

                if isinstance(result, tuple) and len(result) == 2:
                    # If successful, append the plaintext and trace
                    pt, tr = result
                    pts.append(pt)
                    traces.append(tr)
                    successful_traces_count += 1
                    # Provide periodic updates on progress
                    if successful_traces_count % 100 == 0:
                        print(f"[*] Collected {successful_traces_count}/{n} traces successfully...")
                else:
                    # If collection failed (result is an error string), log the failure
                    print(f"[!] Trace request ID {original_trace_id}: {result}")

                # If more successful traces are needed AND we haven't hit the overall attempt limit,
                # submit a new task to keep the worker pool busy.
                if successful_traces_count < n and total_attempts_made < max_total_attempts:
                    new_future = executor.submit(collect_single_trace, current_trace_id)
                    futures[new_future] = current_trace_id
                    current_trace_id += 1
                    total_attempts_made += 1
                elif successful_traces_count < n and total_attempts_made >= max_total_attempts:
                    # If total attempts maxed out before collecting 'n' traces, stop.
                    print(f"[!] Max total collection attempts ({max_total_attempts}) reached. Stopping trace collection.")
                    futures.clear() # Clear remaining futures to break out of outer loop
                    break # Exit current loop

            # If no more futures are active but we haven't reached 'n' successful traces,
            # it means we're stuck or all possible attempts have been made.
            if not futures and successful_traces_count < n:
                print(f"[!] All submitted trace requests have completed, but only {successful_traces_count}/{n} traces were collected successfully.")
                break # Exit the main while loop

    # Final report on collected traces
    if len(pts) < n:
        print(f"[!] Final count: Only {len(pts)}/{n} traces collected. This may affect key recovery accuracy.")
    return pts, traces


# --- Lascar Data Getters ---
# These classes are custom implementations required by lascar's AcquisitionFromGetters.
# They manage their own internal index to sequentially provide plaintext values and leakage traces.

class ValueGetter:
    """
    Provides plaintext values to lascar's AcquisitionFromGetters.
    
    It maintains an internal index to iterate through the list of collected plaintexts.
    If lascar requests a value beyond the available data, it returns a dummy array
    to prevent crashes, ensuring lascar's internal processes can complete gracefully.
    """
    def __init__(self, values):
        self.values = values
        self.current_idx = 0 # Initialize internal index for sequential access
        self.dummy_value_shape = (16,) # Plaintext is always 16 bytes
        self.dummy_value_dtype = np.uint8
        
    def get(self):
        """
        Returns the next plaintext in the sequence as a NumPy array.
        """
        if self.current_idx < len(self.values):
            val_bytes = self.values[self.current_idx]
            val_np_array = np.frombuffer(val_bytes, dtype=self.dummy_value_dtype)
            self.current_idx += 1
            return val_np_array # Return the actual NumPy array
        # Fallback: If out of bounds, return a dummy array to prevent AttributeError/IndexError
        return np.zeros(self.dummy_value_shape, dtype=self.dummy_value_dtype)

class LeakageGetter:
    """
    Provides leakage traces to lascar's AcquisitionFromGetters.
    
    It maintains an internal index to iterate through the list of collected traces.
    If lascar requests a trace beyond the available data, it returns a dummy array
    to prevent crashes, ensuring lascar's internal processes can complete gracefully.
    """
    def __init__(self, traces, dummy_trace_shape, dummy_trace_dtype):
        self.traces = traces
        self.current_idx = 0 # Initialize internal index for sequential access
        self.dummy_trace_shape = dummy_trace_shape # Shape of a single trace (e.g., (1042,))
        self.dummy_trace_dtype = dummy_trace_dtype # Data type of trace samples (e.g., float64)
        
    def get(self):
        """
        Returns the next leakage trace in the sequence as a NumPy array.
        """
        if self.current_idx < len(self.traces):
            trace = self.traces[self.current_idx]
            self.current_idx += 1
            return trace # Return the actual NumPy array
        # Fallback: If out of bounds, return a dummy array to prevent AttributeError/IndexError
        return np.zeros(self.dummy_trace_shape, dtype=self.dummy_trace_dtype)

# --- Lascar CPA Session Execution ---
def run_lascar_session(plaintexts, traces):
    """
    Executes the Correlation Power Analysis (CPA) using the collected data.
    
    This function sets up the lascar acquisition, defines the power model,
    configures the CPA engine, and runs the attack for each byte of the AES key.
    
    Args:
        plaintexts (list): A list of NumPy arrays, where each array is a plaintext.
        traces (list): A list of NumPy arrays, where each array is a power trace.
        
    Returns:
        bytes: The recovered AES key (16 bytes). Returns None if CPA fails.
    """
    # Basic validation: ensure data is available and consistent
    if not plaintexts or not traces or len(plaintexts) != len(traces):
        print("[!] No valid plaintexts or traces available for CPA. Aborting.")
        return None
    
    # Determine the shape and data type of a single power trace.
    # This is crucial for lascar's internal setup (e.g., allocating memory for correlation arrays).
    # We loop to find the first non-empty trace to get accurate shape information.
    first_trace_shape = None
    first_trace_dtype = None
    for tr in traces:
        if tr is not None and tr.size > 0:
            first_trace_shape = tr.shape
            first_trace_dtype = tr.dtype
            break

    if first_trace_shape is None:
        print("[!] No valid non-empty traces collected to determine samples_shape/dtype. Aborting CPA.")
        return None

    key_guess = []
    print("[*] Starting CPA analysis for each key byte...")
    
    # Iterate through each of the 16 bytes of the AES key
    for byte in range(16):
        # Create FRESH instances of ValueGetter and LeakageGetter for EACH byte.
        # This is a crucial workaround for potential internal state management issues
        # or early consumption of getter values by lascar's AcquisitionFromGetters
        # during its initialization for each new CPA engine/session for a byte.
        # It ensures that each byte's CPA analysis starts with a fresh data stream.
        acquisition = AcquisitionFromGetters(
            number_of_traces=len(traces),              # Total number of traces available
            value_getter=ValueGetter(plaintexts),      # Instance to get plaintexts
            leakage_getter=LeakageGetter(traces, first_trace_shape, first_trace_dtype), # Instance to get traces, with shape/dtype for dummy arrays
            # Removed samples_shape and samples_dtype directly from AcquisitionFromGetters
            # This allows lascar to infer them from the first samples provided by the getters,
            # or from the `leakage_getter`'s internal dummy shape if needed.
        )

        def selection_func(value, guess, b=byte):
            """
            Defines the power model (hypothesized leakage for a given key guess).
            
            This function calculates the Hamming Weight of the AES S-box output
            for a specific plaintext byte and a guessed key byte. This is the core
            of the CPA attack, linking power consumption to cryptographic operations.
            
            Args:
                value (numpy.ndarray): The current plaintext as a NumPy array (from ValueGetter).
                guess (int): The current key byte guess (0-255).
                b (int): The index of the current byte being attacked (0-15).
                
            Returns:
                int: The Hamming Weight, representing the expected power leakage.
            """
            # value[b] accesses the specific byte of the plaintext.
            # sbox[plaintext_byte ^ guess] applies the S-box transformation.
            # HW[...] then gets the Hamming Weight of that result.
            return HW[sbox[value[b] ^ guess]]

        # Initialize the CPA engine for the current byte.
        # The 'selection_function' defines the power model, and 'guess_range' specifies
        # the possible values for the key byte (0-255).
        # `jit=True` enables Numba JIT compilation for performance.
        engine = CpaEngine(
            name=f"cpa_byte_{byte}", 
            selection_function=selection_func, 
            guess_range=range(256),
            jit=True 
        )
        
        # Create a lascar session to run the CPA attack for this byte.
        # It links the acquisition data with the CPA engine.
        session = Session(acquisition, engines=[engine])
        
        # Run the CPA attack. Batches process traces in chunks to manage memory.
        session.run(batch_size=100) 

        # Finalize the engine to get the correlation results (e.g., Pearson correlation coefficients).
        results = engine.finalize()

        # Find the best guess: The key byte guess that yields the maximum absolute correlation.
        # The peak in correlation typically indicates the correct key byte.
        best_guess = np.argmax(np.max(np.abs(results), axis=1))
        print(f"[Byte {byte:02d}] Best Guess: {hex(best_guess)}")
        key_guess.append(best_guess) # Add the best guess for this byte to the overall key
    
    return bytes(key_guess) # Return the full recovered key

# --- Main Execution Block ---
if __name__ == "__main__":
    print("[*] Starting DPA script...")
    # Step 1: Collect power traces from the remote server.
    # We aim for 1000 traces, using 20 workers for parallelism,
    # and allow up to 5 times more attempts than successful traces needed (1000 * 5 = 5000 total attempts)
    # to account for network flakiness.
    pts, trs = collect_traces_parallel(n=1000, workers=20, max_overall_attempts_factor=5)

    if pts and trs:
        print(f"[+] Successfully collected {len(pts)} traces. Proceeding with CPA.")
        # Step 2: Run the CPA session to recover the key using the collected data.
        recovered_key = run_lascar_session(pts, trs)

        if recovered_key:
            print("\n[+] Recovered Key:", recovered_key.hex())
            print("[*] Verifying recovered key with server...")
            # Step 3: Send the recovered key to the server for final verification.
            # Option '2' is typically used for key submission.
            response = interact_with_server(b'2', recovered_key.hex().encode())
            if response:
                print("[+] Server Response (Flag/Verification):", response.decode('ascii', errors='ignore'))
            else:
                print("[-] Failed to get a response from the server during key verification.")
        else:
            print("[-] Key recovery failed due to insufficient or problematic trace data. Cannot verify.")
    else:
        print("[-] Trace collection failed. Cannot proceed with DPA.")
    print("[*] Script execution finished.")


