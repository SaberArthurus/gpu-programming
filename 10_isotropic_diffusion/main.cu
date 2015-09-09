// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2015, September 7 - October 6
// ###
// ###
// ### Thomas Moellenhoff, Robert Maier, Caner Hazirbas
// ###
// ###
// ###
// ### THIS FILE IS SUPPOSED TO REMAIN UNCHANGED
// ###
// ###


#include "aux.h"
#include <iostream>
#include <cmath>
using namespace std;


#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8

#define EPSILON 0.01
// uncomment to use the camera
//#define CAMERA

__device__ float g_huber (float gradNorm)
{
    float eps = EPSILON;
    return 1.0f / (max(eps, gradNorm));
}

__device__ float g_exp (float gradNorm)
{
    float eps = EPSILON;
    return exp(- gradNorm * gradNorm / eps) / eps;
}

__global__ void compute_gradient (float *d_imgIn, float *d_gradH, float *d_gradV, int w, int h, int nc)
{
    // d_imgIn, d_gradH and d_gradV are (w * h) * nc images
    // this kernel calculated the derivatives of each pixel in hor and vert directions
    // and stores them to d_gradH and d_gradV arrays

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
    }
}

__global__ void make_time_step (float *d_imgIn, float *d_gradH, float *d_gradV, int w, int h, int nc, float timeStep, char diffType)
{
    // this kernel computes diffusivity g, div(g * grad) and updates the input images
    size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    size_t y = threadIdx.y + blockIdx.y * blockDim.y;


    float g;     // diffusivity 

    if (x < w and y < h)
    {
        // 1-D index in the flattened array
        size_t ind = x + y * w;
        
   
        // First, find the value of g
        // To find g, compute the norm of the gradient
        float gradNorm = 0;
        for (int chan = 0; chan < nc; chan++)
        {
            int chan_offset = chan * w * h;

            gradNorm += d_gradH[ind + chan_offset] * d_gradH[ind + chan_offset];
            gradNorm += d_gradV[ind + chan_offset] * d_gradV[ind + chan_offset];
        }    
        
        gradNorm = sqrt(gradNorm);

        switch (diffType)
        {
            case 'l':
                g = 1;
                break;
            case 'h':
                g = g_huber(gradNorm);
                break;
            case 'e':
                g = g_exp(gradNorm);
                break;
        }

        // Now calculate the divergence and update the image
        for (int chan = 0; chan < nc; chan++)
        {
            int chan_offset = chan * w * h;

            // Compute divergence of the gradient using backward differences
            bool isBoundary = (x == 0); 
            float horGrad = (isBoundary ? g * d_gradH[ind + chan_offset] : g * (d_gradH[ind + chan_offset] - d_gradH[ind - 1 + chan_offset]));

            isBoundary = (y == 0);
            float verGrad = (isBoundary ? g * d_gradV[ind + chan_offset] : g * (d_gradV[ind + chan_offset] - d_gradV[ind - w + chan_offset]));

            d_imgIn[ind + chan_offset] += timeStep * (horGrad + verGrad);
        }

    }


}

int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;




    // Reading command line parameters:
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
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray] [-n <numSteps>] [-step <timeStep>]" << endl; return 1; }
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
    
    // size of the time step
    float timeStep = 0.1;
    getParam("step", timeStep, argc, argv);

    cout << "Diffusion for " << numSteps << " steps with timestep = " << timeStep << endl; 

    // type of diffusion
    char diffType = 'l';
    getParam("type", diffType, argc, argv);
   

    // Init camera / Load input image
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




    // Set the output image format
    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    //cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    // ### Define your own output images here as needed




    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn  = new float[(size_t)w*h*nc];

    // // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    // float *imgOut = new float[(size_t)w*h*mOut.channels()];

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
    cudaMalloc(&d_imgIn, w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_gradH, w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_gradV, w * h * nc * sizeof(float)); CUDA_CHECK;

    cudaMemcpy(d_imgIn, imgIn, w * h * nc * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;


    dim3 block = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1); 
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);


    Timer timer; timer.start();
    for (int i = 0; i < numSteps; i++)
    {
        compute_gradient <<<grid, block>>> (d_imgIn, d_gradH, d_gradV, w, h, nc);
        make_time_step <<<grid, block>>> (d_imgIn, d_gradH, d_gradV, w, h, nc, timeStep, diffType);
    }
    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "time: " << t*1000 << " ms" << endl;

    cudaMemcpy(imgIn, d_imgIn, w * h * nc * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

    // Free device arrays
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_gradH); CUDA_CHECK;
    cudaFree(d_gradV); CUDA_CHECK;




    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgIn);
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



