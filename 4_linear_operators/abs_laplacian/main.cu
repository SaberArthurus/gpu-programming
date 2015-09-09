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

// uncomment to use the camera
//#define CAMERA



__global__ void gradient (float *d_imgIn, float *d_gradH, float *d_gradV, int w, int h, int nc)
{
    size_t ind = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    if (ind < w * h * nc)
    {   
        // Gradient in horizontal direction
        bool isBoundary = (ind % w == w - 1);  
        d_gradH[ind] = (isBoundary ? 0 : (d_imgIn[ind + 1] - d_imgIn[ind]));

        // Gradient in vertical direction
        isBoundary = (ind % (w * h) >= (w * (h - 1)));
        d_gradV[ind] = (isBoundary ? 0 : (d_imgIn[ind + w] - d_imgIn[ind]));
    }
}

__global__ void divergence (float *d_imgHGrad, float *d_imgVGrad, float *d_imgLapl, int w, int h, int nc)
{
    size_t ind = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    if (ind < w * h * nc)
    {   
        // Backward difference gradient in horizontal direction
        bool isBoundary = (ind % w == 0); 
        float horGrad = (isBoundary ? 0 : (d_imgHGrad[ind] - d_imgHGrad[ind - 1]));

        // Backward difference gradient in vertical direction
        isBoundary = (ind % (w * h) < w);
        float verGrad = (isBoundary ? 0 : (d_imgVGrad[ind ] - d_imgVGrad[ind - w]));

        d_imgLapl[ind] = horGrad + verGrad;
    }
}

__global__ void l2_norm (float *d_imgLapl, float *d_imgAbsLapl, int w, int h, int nc)
{
    size_t ind = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    if (ind < w * h)
    {   
        d_imgAbsLapl[ind] = 0;
        for (int c = 0; c < nc; c++)
        {
            d_imgAbsLapl[ind] += d_imgLapl[ind + c * w * h] * d_imgLapl[ind + c * w * h];
        }
        d_imgAbsLapl[ind] = sqrt(d_imgAbsLapl[ind]);
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
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image>  -g <gamma>[-repeats <repeats>] [-gray]" << endl; return 1; }
#endif
    
    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;


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
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    //cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    // ### Define your own output images here as needed

    // Matrix for absolute of Laplacial
    cv::Mat mOut(h, w, CV_32FC1);


    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn  = new float[(size_t)w * h * nc];

    // allocate raw output arrays for all the intermediate values
    float *imgOut = new float[(size_t)w * h];


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


    // MAIN COMPUTATION

    // Init image array on the device
    float *d_imgIn = NULL;
    float *d_imgHGrad = NULL;
    float *d_imgVGrad = NULL;
    float *d_imgLapl = NULL;
    float *d_imgAbsLapl = NULL;
    cudaMalloc(&d_imgIn, w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_imgHGrad, w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_imgVGrad, w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_imgLapl, w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_imgAbsLapl, w * h * sizeof(float)); CUDA_CHECK;

    // move from host to device memory
    cudaMemcpy(d_imgIn, imgIn, w * h * nc * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

    dim3 block;
    dim3 grid;

    Timer timer; timer.start();

    for (int rep = 0; rep < repeats; rep++)
    {
        // initialize block and grid size
        block = dim3(64, 1, 1); 
        grid = dim3((w * h * nc + block.x * block.y * block.z - 1) / (block.x * block.y * block.z), 1, 1);

        gradient <<<grid, block>>> (d_imgIn, d_imgHGrad, d_imgVGrad, w, h, nc);
        divergence <<<grid, block>>> (d_imgHGrad, d_imgVGrad, d_imgLapl, w, h, nc);

        // adjust the grid size because now we only need 1 kernel per pixel for ALL the channels
        grid = dim3((w * h + block.x * block.y * block.z - 1) / (block.x * block.y * block.z), 1, 1);
        l2_norm <<<grid, block>>> (d_imgLapl, d_imgAbsLapl, w, h, nc);
    }
    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "average kernel time: " << t*1000/repeats << " ms" << endl;

    // copy result back to host memory
    cudaMemcpy(imgOut, d_imgAbsLapl, w * h * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

    // free the device memory
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_imgHGrad); CUDA_CHECK;
    cudaFree(d_imgVGrad); CUDA_CHECK;
    cudaFree(d_imgLapl); CUDA_CHECK;
    cudaFree(d_imgAbsLapl); CUDA_CHECK;

    // // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // // show output image: first convert to interleaved opencv format from the layered raw array
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
    cv::imwrite("image_input.png", mIn * 255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_output.png", mOut * 255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



