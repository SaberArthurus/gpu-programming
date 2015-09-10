// ### Practical Course: GPU Programming in Computer Vision
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2015, September 7 - October 6
// ###
// ### Thomas Moellenhoff, Robert Maier, Caner Hazirbas
// ### 

// ### Assignment 10 - Isotropic diffusion
// ###
// ### The program performs the isotropic diffusion iteration on the <image> for a total of <numSteps> steps of size <timeStep>
// ### 
// ### Usage: "./main -i <image> [-repeats <repeats>] [-gray] [-n <numSteps>] [-step <timeStep>]"
// ### diffType type can be l - linear, h - Hubert (g = 1 / max(eps, s)), e - exponential (g = exp(-s^2/eps)/eps)
// ###
// ### Size of the block is defined by BLOCK_SIZE_X, BLOCK_SIZE_Y
// ### 

#include "aux.h"
#include <iostream>
#include <cmath>
using namespace std;


#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8

#define EPSILON 0.01

// Choice of iteration for red-black Gauss-Seidel
#define RED 0
#define BLACK 1

// uncomment to use the camera
//#define CAMERA

__device__ float compute_g (float gradNorm)
{
    float eps = EPSILON;
    return 1.0f / (max(eps, gradNorm));
}


__global__ void compute_gradient_and_diffusivity (float *d_imgIn, float *d_gradH, float *d_gradV, float *d_g, int w, int h, int nc)
{
    // d_imgIn, d_gradH and d_gradV are (w * h) * nc images
    // this kernel calculated the derivatives of each pixel in hor and vert directions,
    // stores them to d_gradH and d_gradV arrays, and computes diffusivity (g) for each pixel

    size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    size_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < w and y < h)
    {   
        // 1-D index in the flattened array
        size_t ind = x + y * w;

        // All channels are updates within single kernel
        for (int chan = 0; chan < nc; chan++)
        {
            size_t chan_offset = chan * w * h;

            // Compute gradient using forward differences
            // grad is 0 on right and bottom boundaries
            bool isBoundary = (x == w - 1);  
            d_gradH[ind + chan_offset] = (isBoundary ? 0 : (d_imgIn[ind + 1 + chan_offset] - d_imgIn[ind + chan_offset]));

            isBoundary = (y == h - 1);
            d_gradV[ind + chan_offset] = (isBoundary ? 0 : (d_imgIn[ind + w + chan_offset] - d_imgIn[ind + chan_offset]));
        }

        // Now we can compute g (diffusivity)
        // To find g, compute the norm of the gradient
        float gradNorm = 0;
        for (int chan = 0; chan < nc; chan++)
        {
            int chan_offset = chan * w * h;

            gradNorm += d_gradH[ind + chan_offset] * d_gradH[ind + chan_offset];
            gradNorm += d_gradV[ind + chan_offset] * d_gradV[ind + chan_offset];
        }    
        
        gradNorm = sqrt(gradNorm);

        d_g[x + y * w] = compute_g(gradNorm);
    }
}

__global__ void perform_update (float *d_imgIn, float *d_solOld, float *d_gradH, float *d_gradV, float *d_g, float *d_solNew, 
                                int w, int h, int nc, float lambda, float theta, int color)
{
    // this kernel computes diffusivity g in each direction and performs one step of Jacobi iteration
    size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    size_t y = threadIdx.y + blockIdx.y * blockDim.y;


    if (x < w and y < h)
    {
        // 1-D index in the flattened array
        size_t ind = x + y * w;
        
        // Check if this pixel is to be updated on this iteration
        if ((x + y) % 2 == color)
        {
            // Find the values of diffusivity in each direction if they exist
            int g_r, g_l, g_u, g_d;
            g_r = (x + 1 < w) * d_g[x + y * w];
            g_l = (x > 0) * d_g[x - 1 + y * w];
            g_u = (y + 1 < h) * d_g[x + y * w];
            g_d = (y > 0) * d_g[x + (y - 1) * w];

            // Now calculate the divergence and perform one update
            for (int chan = 0; chan < nc; chan++)
            {
                int chan_offset = chan * w * h;
                // Compute the value of the update for specific channel
                float update = (2 * d_imgIn[ind + chan_offset] + lambda * g_r * d_solOld[ind + 1 + chan_offset] + 
                        lambda * g_l * d_solOld[ind - 1 + chan_offset] + lambda * g_u * d_solOld[ind + w + chan_offset] + 
                        lambda * g_d * d_solOld[ind - w + chan_offset]) / (2 + lambda * (g_r + g_l + g_u + g_d));
                d_solNew[ind + chan_offset] = update + theta * (update - d_solOld[ind + chan_offset]);
            }
        }

    }


}

int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;




    // ---------------------------READING COMMAND LINE PARAMETERS--------------------------
    // getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
    // If "-param" is not specified, the value of "var" remains unchanged
    //
    // return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise

#ifdef CAMERA
#else
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << 
        " -i <image> [-repeats <repeats>] [-gray] [-n <numSteps>] [-step <timeStep>]" << endl; return 1; }
#endif
    
    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // number of timeSteps to take
    int numSteps = 10;
    getParam("n", numSteps, argc, argv);
    
    // regularization parameter
    float lambda = 0.1;
    getParam("lambda", lambda, argc, argv);

    // successive over relaxation parameter
    float theta = 0.5;
    getParam("theta", theta, argc, argv);


    cout << "Denoising with N = " << numSteps << ", lambda = " << lambda << ", theta = " << theta << endl;
   


    // ---------------------------INIT CAMERA / LOAD INPUT IMAGE---------------------------
#ifdef CAMERA

    // Init camera
  	cv::VideoCapture camera(0);
  	if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;
  	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions
    cv::Mat mIn;
    camera >> mIn;
    
#else
    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
    
#endif

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    cout << "image: " << w << " x " << h << endl;

    // ----------------------------------APLLY BLUR TO INPUT IMAGE----------------------------------
    addNoise(mIn, 0.1f);


    // Set the output image format
    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    //cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer




    // ----------------------------------ALLOCATE ARRAYS ON HOST----------------------------------
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn  = new float[(size_t)w*h*nc];

    // // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];

    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    while (cv::waitKey(30) < 0)
    {
    // Get camera image
    camera >> mIn;
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
#endif

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);

    // ------------------------------ ALLOCATE ARRAYS ON THE DEVICE -----------------------------
    float *d_imgIn = NULL;
    float *d_gradH = NULL;
    float *d_gradV = NULL;
    float *d_g = NULL;
    float *d_solOld = NULL;
    float *d_solNew = NULL;
    cudaMalloc(&d_imgIn, w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_gradH, w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_gradV, w * h * nc * sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_solOld, w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_solNew, w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_g, w * h * sizeof(float)); CUDA_CHECK;

    cudaMemcpy(d_imgIn, imgIn, w * h * nc * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;


    dim3 block = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1); 
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);

    // d_imgIn, d_solOld, d_gradH, d_gradV, d_g, d_solNew
    Timer timer; timer.start();
    for (int i = 0; i < numSteps; i++)
    {
        compute_gradient_and_diffusivity <<<grid, block>>> (d_imgIn, d_gradH, d_gradV, d_g, w, h, nc);
        perform_update <<<grid, block>>> (d_imgIn, d_solOld, d_gradH, d_gradV, d_g, d_solNew, w, h, nc, lambda, theta, RED);
        cudaMemcpy(d_solOld, d_solNew, w * h * nc * sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
        perform_update <<<grid, block>>> (d_imgIn, d_solOld, d_gradH, d_gradV, d_g, d_solNew, w, h, nc, lambda, theta, BLACK);
        cudaMemcpy(d_solOld, d_solNew, w * h * nc * sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;
    }
    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;

    cudaMemcpy(imgOut, d_solNew, w * h * nc * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

    // Free device arrays
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_gradH); CUDA_CHECK;
    cudaFree(d_gradV); CUDA_CHECK;
    cudaFree(d_solOld); CUDA_CHECK;
    cudaFree(d_g); CUDA_CHECK;
    cudaFree(d_solNew); CUDA_CHECK;




    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mOut, 100+w+40, 100);

    // ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif




    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    // delete[] imgOut;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



