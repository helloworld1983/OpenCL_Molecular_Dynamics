#include "headers.h"
#include "md.cpp"

#ifdef ALTERA
    #include "AOCL_Utils.h"
    using namespace aocl_utils;
#else
    #include "CL/cl.h"
    void checkError(cl_int err, const char *operation){
        if (err != CL_SUCCESS){
            fprintf(stderr, "Error during operation '%s': %d\n", operation, err);
            exit(1);
        }
    }
#endif
#ifdef NVIDIA
    #define VENDOR "NVIDIA Corporation"
#endif
#ifdef IOCL
    #define VENDOR "Intel(R) Corporation"
#endif

cl_platform_id platform = NULL;
cl_device_id device;
cl_context context = NULL;
cl_command_queue queue;
cl_program program = NULL;
cl_kernel kernel;
cl_mem nearest_buf;
cl_mem output_energy_buf;
cl_mem output_force_buf;
cl_mem charge_buf;

cl_float3 position_arr[particles_count] = {};
cl_float3 nearest[particles_count] = {};
cl_float3 velocity[particles_count] = {};
cl_int charge[particles_count] = {};

cl_float output_energy[particles_count] = {};
cl_float3 output_force[particles_count] = {};
double kernel_total_time = 0.;
cl_float final_energy = 0.;
bool (*init_opencl)() = init_opencl_lj;
void (*run)() = run_lj;

int main(int argc, char *argv[]) {
    struct timeb start_total_time;
    ftime(&start_total_time);
    if (argc > 1){
        if (!strcmp(argv[1], COULOMB)){
            init_opencl = init_opencl_coulomb;
            run = run_coulomb;
        }
        else{
        	if (!strcmp(argv[1], "--help")){
        		printf("Usage: %s [--help][--coulomb]", argv[0]);
        	}
        	else{
        		printf("invalid argument\n");
        		printf("Usage: %s [--help][--coulomb]", argv[0]);
        	}
        }
    }
    if(!init_opencl()) {
      return -1;
    }
    init_problem(position_arr, velocity, charge);
    md(position_arr, nearest, output_force, output_energy, velocity, charge);
    cleanup();
    struct timeb end_total_time;
    ftime(&end_total_time);
    printf("Total execution time in ms =  %d\n", (int)((end_total_time.time - start_total_time.time) * 1000 + end_total_time.millitm - start_total_time.millitm));
    printf("Kernel execution time in milliseconds = %0.3f ms\n", (kernel_total_time / 1000000.0) );
    printf("Kernel execution time in milliseconds per iters = %0.3f ms\n", (kernel_total_time / ( total_it * 1000000.0)) );
    printf("energy is %f \n",final_energy);
    return 0;
}

/////// HELPER FUNCTIONS ///////

// Initializes the OpenCL objects.
bool init_opencl_lj() {
    cl_int status;

    printf("Initializing OpenCL\n");
    #ifdef ALTERA
        if(!setCwdToExeDir()) {
          return false;
        }
        platform = findPlatform("Altera");
    #else
        cl_uint num_platforms;
        cl_platform_id pls[MAX_PLATFORMS_COUNT];
        clGetPlatformIDs(MAX_PLATFORMS_COUNT, pls, &num_platforms);
        char vendor[128];
        for (int i = 0; i < MAX_PLATFORMS_COUNT; i++){
            clGetPlatformInfo (pls[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
            if (!strcmp(VENDOR, vendor))
            {
                platform = pls[i];
                break;
            }
        }
    #endif
    if(platform == NULL) {
      printf("ERROR: Unable to find OpenCL platform.\n");
      return false;
    }

    #ifdef ALTERA
        scoped_array<cl_device_id> devices;
        cl_uint num_devices;
        devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
        // We'll just use the first device.
        device = devices[0];
    #else
        cl_uint num_devices;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU , 1, &device, &num_devices);
    #endif

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    checkError(status, "Failed to create context");

    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    #ifdef ALTERA
        std::string binary_file = getBoardBinaryFile("md_lj", device);
        printf("Using AOCX: %s\n", binary_file.c_str());
        program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);
    #else
        int MAX_SOURCE_SIZE  = 65536;
        FILE *fp;
        FILE *fp2;
        const char fileName[] = "./device/md_lj.cl";
        const char header[] = "./include/parameters.h";
        size_t source_size;
        char *source_str;
        int count = 0;
        try {
            fp = fopen(fileName, "r");
            if (!fp) {
                fprintf(stderr, "Failed to load kernel.\n");
                exit(1);
            }
            fp2 = fopen(header, "r");
            if (!fp2) {
                fprintf(stderr, "Failed to load kernel header.\n");
                exit(1);
            }
            source_str = (char *)malloc(MAX_SOURCE_SIZE);
            char ch = getc(fp2);
            while (ch != EOF){
                source_str[count] = ch;
                count++;
                ch = getc(fp2);
            }
            ch = getc(fp);
            source_str[count++] = '\n';
            int skip_flag = 0;
            while(ch != EOF){
                if (ch == '#'){
                    skip_flag = 1;
                }
                if (ch == '\n'){
                    skip_flag = 0;
                }
                if (!skip_flag){
                    source_str[count] = ch;
                    count++;
                }
                ch = getc(fp);
            }
            fclose(fp);
            fclose(fp2);
        }
        catch (int a) {
            printf("%f", a);
        }
        program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &status);
    #endif

    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    const char *kernel_name = "md";
    kernel = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");

    // Input buffer.
    nearest_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        particles_count * sizeof(cl_float3), NULL, &status);
    checkError(status, "Failed to create buffer for nearest");

    // Output buffers.
    output_energy_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        particles_count * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output_en");

     output_force_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        particles_count * sizeof(cl_float3), NULL, &status);
    checkError(status, "Failed to create buffer for output_force");

    return true;
}

bool init_opencl_coulomb() {
    cl_int status;

    printf("Initializing OpenCL\n");
    #ifdef ALTERA
        if(!setCwdToExeDir()) {
          return false;
        }
        platform = findPlatform("Altera");
    #else
        cl_uint num_platforms;
        cl_platform_id pls[MAX_PLATFORMS_COUNT];
        clGetPlatformIDs(MAX_PLATFORMS_COUNT, pls, &num_platforms);
        char vendor[128];
        for (int i = 0; i < MAX_PLATFORMS_COUNT; i++){
            clGetPlatformInfo (pls[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
            if (!strcmp(VENDOR, vendor))
            {
                platform = pls[i];
                break;
            }
        }
    #endif
    if(platform == NULL) {
      printf("ERROR: Unable to find OpenCL platform.\n");
      return false;
    }

    #ifdef ALTERA
        scoped_array<cl_device_id> devices;
        cl_uint num_devices;
        devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
        // We'll just use the first device.
        device = devices[0];
    #else
        cl_uint num_devices;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU , 1, &device, &num_devices);
    #endif

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    checkError(status, "Failed to create context");

    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    #ifdef ALTERA
        std::string binary_file = getBoardBinaryFile("md_coulomb", device);
        printf("Using AOCX: %s\n", binary_file.c_str());
        program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);
    #else
        int MAX_SOURCE_SIZE  = 65536;
        FILE *fp;
        FILE *fp2;
        const char fileName[] = "./device/md_coulomb.cl";
        const char header[] = "./include/parameters.h";
        size_t source_size;
        char *source_str;
        int count = 0;
        try {
            fp = fopen(fileName, "r");
            if (!fp) {
                fprintf(stderr, "Failed to load kernel.\n");
                exit(1);
            }
            fp2 = fopen(header, "r");
            if (!fp2) {
                fprintf(stderr, "Failed to load kernel header.\n");
                exit(1);
            }
            source_str = (char *)malloc(MAX_SOURCE_SIZE);
            char ch = getc(fp2);
            while (ch != EOF){
                source_str[count] = ch;
                count++;
                ch = getc(fp2);
            }
            ch = getc(fp);
            source_str[count++] = '\n';
            int skip_flag = 0;
            while(ch != EOF){
                if (ch == '#'){
                    skip_flag = 1;
                }
                if (ch == '\n'){
                    skip_flag = 0;
                }
                if (!skip_flag){
                    source_str[count] = ch;
                    count++;
                }
                ch = getc(fp);
            }
            fclose(fp);
            fclose(fp2);
        }
        catch (int a) {
            printf("%f", a);
        }
        program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &status);
    #endif

    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    const char *kernel_name = "md";
    kernel = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");

    // Input buffer.
    nearest_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        particles_count * sizeof(cl_float3), NULL, &status);
    checkError(status, "Failed to create buffer for nearest");

    //charge buffer
    charge_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        particles_count * sizeof(cl_int), NULL, &status);
    checkError(status, "Failed to create buffer for charge");

    // Output buffers.
    output_energy_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        particles_count * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output_en");

     output_force_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        particles_count * sizeof(cl_float3), NULL, &status);
    checkError(status, "Failed to create buffer for output_force");

    return true;
}

void run_lj() {
    cl_int status;

    cl_event kernel_event;
    cl_event finish_event[2];
    cl_ulong time_start, time_end;
    double total_time;

    cl_event write_event;
    status = clEnqueueWriteBuffer(queue, nearest_buf, CL_FALSE,
        0, particles_count * sizeof(cl_float3), nearest, 0, NULL, &write_event);
    checkError(status, "Failed to transfer nearest");

    unsigned argi = 0;

    size_t global_work_size[1] = {particles_count};
    size_t local_work_size[1] = {particles_count};
    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &nearest_buf);
    checkError(status, "Failed to set argument nearest");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_energy_buf);
    checkError(status, "Failed to set argument output_energy");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_force_buf);
    checkError(status, "Failed to set argument output_force");

    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
        global_work_size, local_work_size, 1, &write_event, &kernel_event);
    checkError(status, "Failed to launch kernel");

    status = clEnqueueReadBuffer(queue, output_energy_buf, CL_FALSE,
        0, particles_count * sizeof(float), output_energy, 1, &kernel_event, &finish_event[0]);

    status = clEnqueueReadBuffer(queue, output_force_buf, CL_FALSE,
        0, particles_count * sizeof(cl_float3), output_force, 1, &kernel_event, &finish_event[1]);

    // Release local events.
    clReleaseEvent(write_event);

    // Wait for all devices to finish.
    clWaitForEvents(2, finish_event);

    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    total_time = time_end - time_start;
    kernel_total_time += total_time;

    // Release all events.
    clReleaseEvent(kernel_event);
    clReleaseEvent(finish_event[0]);
    clReleaseEvent(finish_event[1]);
}

void run_coulomb() {
    cl_int status;

    cl_event kernel_event;
    cl_event finish_event[2];
    cl_ulong time_start, time_end;
    double total_time;

    cl_event write_event[2];
    status = clEnqueueWriteBuffer(queue, nearest_buf, CL_FALSE,
        0, particles_count * sizeof(cl_float3), nearest, 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer nearest");

    status = clEnqueueWriteBuffer(queue, charge_buf, CL_FALSE,
        0, particles_count * sizeof(cl_int), charge, 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer charge");

    unsigned argi = 0;

    size_t global_work_size[1] = {particles_count};
    size_t local_work_size[1] = {particles_count};
    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &nearest_buf);
    checkError(status, "Failed to set argument nearest");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &charge_buf);
    checkError(status, "Failed to set argument charge");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_energy_buf);
    checkError(status, "Failed to set argument output_energy");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_force_buf);
    checkError(status, "Failed to set argument output_force");

    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
        global_work_size, local_work_size, 1, write_event, &kernel_event);
    checkError(status, "Failed to launch kernel");

    status = clEnqueueReadBuffer(queue, output_energy_buf, CL_FALSE,
        0, particles_count * sizeof(float), output_energy, 1, &kernel_event, &finish_event[0]);

    status = clEnqueueReadBuffer(queue, output_force_buf, CL_FALSE,
        0, particles_count * sizeof(cl_float3), output_force, 1, &kernel_event, &finish_event[1]);

    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);

    // Wait for all devices to finish.
    clWaitForEvents(2, finish_event);

    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    total_time = time_end - time_start;
    kernel_total_time += total_time;

    // Release all events.
    clReleaseEvent(kernel_event);
    clReleaseEvent(finish_event[0]);
    clReleaseEvent(finish_event[1]);
}

// Free the resources allocated during initialization
void cleanup() {
    if(kernel) {
      clReleaseKernel(kernel);
    }
    if(queue) {
      clReleaseCommandQueue(queue);
    }
    if(nearest_buf) {
      clReleaseMemObject(nearest_buf);
    }
    if(charge_buf){
        clReleaseMemObject(charge_buf);
    }
    if(output_energy_buf) {
      clReleaseMemObject(output_energy_buf);
    }
    if(output_force_buf) {
      clReleaseMemObject(output_force_buf);
    }
    if(program) {
    clReleaseProgram(program);
    }
    if(context) {
    clReleaseContext(context);
    }
}

