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

#define _USE_MATH_DEFINES
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8
// uncomment to use the camera
//#define CAMERA

texture<float, 2, cudaReadModeElementType> texRef;

__global__ void perform_convolution (float *d_imgIn, float *d_imgKern, float *d_imgOut, int w, int h, int nc, int r, int w_kernel, int h_kernel)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < w and y < h)
    {
        for (int c = 0; c < nc; c++)
        {
            float accumulated = 0;
            for (int a = -r; a <= r; a++)
            {
                for (int b = -r; b <= r; b++)
                {
                    int x_act = min(max(x + a, 0), w - 1);
                    int y_act = min(max(y + b, 0), h - 1);
                    accumulated += tex2D(texRef, x_act + 0.5f, y_act + c * h + 0.5f) * d_imgKern[(r + a) + (r + b) * w_kernel];
                }
            }
            d_imgOut[x + y * w + c * w * h] = accumulated;
            // d_imgOut[x + y * w + c * w * h] = tex2D(texRef, x + 0.5f, y + h * c + 0.5f);
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
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }
#endif
    
    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // standard deviation of the Gaussian
    float sigma = 1;
    getParam("sigma", sigma, argc, argv);
    cout << "sigma: " << sigma << endl;
    sigma = abs(sigma);

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
    // ###
    // ###
    // ### TODO: Change the output image format as needed
    // ###
    // ###
    cv::Mat mOut(h, w, mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    //cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer

    // Kernel matrix
    int r = (int)ceil(3 * sigma);
    int w_kernel = 2 * r + 1;
    int h_kernel = 2 * r + 1;
    int w_mid = r + 1;
    int h_mid = r + 1;
    cv::Mat mKernel = cv::Mat::zeros(h_kernel, w_kernel, CV_32FC1);

    // Normalize the kernel so that it sums up to 1
    float val = 0;

    for (int i = 0; i < w_kernel; i++)
    {
        for (int j = 0; j < h_kernel; j++)
        {
            val = 1.0 / (2.0 * M_PI * sigma * sigma) * exp(-(pow(i - w_mid, 2) + pow(j - h_mid, 2)) / (2.0 * sigma * sigma));
            mKernel.at<float>(i, j) = val;
        }
    }

    mKernel /= cv::sum(mKernel)[0];

    // Create kernel for visualization, that has max value of 1

    double minV, maxV;
    cv::Point minL, maxL;
    cv::minMaxLoc(mKernel, &minV, &maxV, &minL, &maxL);
    cv::Mat visKernel = mKernel / maxV;

    // Display the visualization kernel
    // showImage("Kernel", visKernel, 100 + w + 40,  100 + h + 40);  // show at position (x_from_left=100,y_from_above=100)
    // cv::waitKey(0);


    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn  = new float[(size_t)w*h*nc];

    // allocate the linearized kernel array
    float *imgKern = new float[w_kernel * h_kernel];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
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
    convert_mat_to_layered (imgKern, mKernel);
    
    // Initialize the arrays on the device    
    float *d_imgIn = NULL;
    float *d_imgKern = NULL;
    float *d_imgOut = NULL;
    cudaMalloc(&d_imgIn, w * h * nc * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_imgKern, w_kernel * h_kernel * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_imgOut, w * h * nc * sizeof(float)); CUDA_CHECK;

    // Allocate the texture
    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.filterMode = cudaFilterModeLinear;
    texRef.normalized = false;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();


    // Move input img and kernel to the device
    cudaMemcpy(d_imgIn, imgIn, w * h * nc * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(d_imgKern, imgKern, w_kernel * h_kernel * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaBindTexture2D(NULL, &texRef, d_imgIn, &desc, w, h * nc, w * sizeof(d_imgIn[0]));
    
    dim3 block = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1); 
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);

    cout << grid.x << endl;
    Timer timer; timer.start();
    for (int i = 0; i < repeats; i++)
    {
        perform_convolution <<<grid, block>>> (d_imgIn, d_imgKern, d_imgOut, w, h, nc, r, w_kernel, h_kernel);
    }
    timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "Average time over " << repeats << " runs using texture: " << t * 1000.0 / repeats << " ms" << endl;
    
    cudaMemcpy(imgOut, d_imgOut, w * h * nc * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_imgKern); CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;
    cudaUnbindTexture(texRef);

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
    delete[] imgOut;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



