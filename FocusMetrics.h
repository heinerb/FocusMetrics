/**
 * Author: Benjamin Kurt Heiner
 *   Date: 7/20/2014
 * heinerb@gmail.com
 **/

#pragma once

/******************************************************************
 * System Include Directives
 *****************************************************************/
#include <stdint.h>						// Allows for the use of uint*_t and etc
#include <map>							// Allows for the use of the std::map and associated containers.
#include <limits>						// Needed for 
#include <cmath>						// used for abs and other items

/******************************************************************
 * OpenCV Include Directives
 *****************************************************************/
#include "opencv2/core/core.hpp"			// Basic include for OpenCV
#include "opencv2/imgproc/imgproc.hpp"		// Image processing
#include "opencv2/highgui/highgui.hpp"		// Image display
#include "opencv2/ocl/ocl.hpp"				// OCL support 
#include "opencv2/gpu/gpu.hpp"				// CUDA support 

/******************************************************************
 * User Include Directives
 *****************************************************************/
#include "ImageFilters.h"					// Common filters used

/******************************************************************
 * Used Namespaces...
 *****************************************************************/
using namespace cv;
using namespace gpu;
using namespace ocl;

/**
 *@brief This class contains Focus metrics used to measure the Focus level in a image. The class contains 6 types of FOCUS metrics which are as follows.
 * 
 * Description from http://www.sayonics.com/publications/pertuz_PR2013.pdf
 * Analysis of focus measure operators for shape-from-focus
 *
 * Matlab port of "Said Pertuz"
 *
 * 1. Gradient-based operators (GRA*). This family groups focus measure operators based on the gradient or first derivative of the image. These algorithms follow the assumption
 * that focused images present more sharp edges than blurredones. Thus, the gradient is used to measure the degree of focus.
 *
 * 2. Laplacian-based operators (LAP*). Similarly to the previous family, the goal of these operators is to measure the amount of edges present in images, although through the
 * second derivative or Laplacian.
 *
 * 3. Wavelet-based operators (WAV*). The focus measure operators within this family take advantage of the capability of the coecients of the discrete wavelet transform to describe
 * the frequency and spatial content of images. Therefore, these coecients can be utilized to measure the focus level.
 *
 * 4. Statistics-based operators (STA*). The focus measure operators within this family take advantage of several image
 * statistics as texture descriptors in order to compute the focus level.
 *
 * 5. DCT-based operators (DCT*). Similarly to the waveletbased operators, this family takes advantage of the discrete cosine transform (DCT) coecients in order to compute
 * the focus level of an image from its frequency content. None of the operators within this family have previously been used in SFF applications to our knowledge.
 *
 * 6. Miscellaneous operators (MIS*). family groups operators that do not belong to any of the previous five groups.
 **/
template<typename MatType>
class FocusMetrics
{

public: // STATIC and Other Variables.

	/**
	 * @brief List of the availible FOCUS metrics.
	 **/
	typedef enum FOCUS_METRIC
	{
		FM_INVALID = -1,
		//http://www.sayonics.com/publications/pertuz_PR2013.pdf
		FM_ACMO,
		FM_BREN,
		FM_CONT,
		FM_CURV,
		FM_DCTE,
		FM_DCTR,
		FM_GDER,
		FM_GLVA,
		FM_GLLV,
		FM_GLVN,
		FM_GRAE,
		FM_GRAT,
		FM_GRAS,
		FM_HELM,
		FM_HISE,
		FM_HISR,
		FM_LAPE,
		FM_LAPM,
		FM_LAPV,
		FM_LAPD,
		FM_SFIL,
		FM_SFRQ,
		FM_TENG,
		FM_TENV,
		FM_VOLA,
		FM_WAVS,
		FM_WAVV,
		FM_WAVR,
		FM_COUNT
	}; 

public:
	/**
	 * @brief computeFocusMetric - This function measures the relative degree of focus of an image. 
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aMetric		The specific metric to compute based on.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric(const MatType & aSrc, const FOCUS_METRIC aMetric,const cv::Size aROI, const int32_t aWindowSize);

private: // Type defines 

	/**
	 * @brief Function pointer for the lookup
	 **/
	typedef double (*FocusMetricPointer) (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize);

	/**
	 * @brief Map type for Focus metric lookup
	 **/
	typedef std::map<FOCUS_METRIC, FocusMetricPointer> FocusMap;

private: 
	static MatType calculateHistogrm(const MatType & aSrc);

private: // Focus metrics from the MIS section --- http://www.sayonics.com/publications/pertuz_PR2013.pdf

	/**
	 * @brief computeFocusMetric_ACMO - This function measures the relative degree of focus of an image using the Absolute Central Moment (Shirvaikar2004) (MIS1). 
	 * http://www.sayonics.com/publications/pertuz_PR2013.pdf
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric_ACMO (const MatType & aSrc,const cv::Size aROI, const int32_t);

	/**
	 * @brief computeFocusMetric_BREN - This function measures the relative degree of focus of an image using the Brenner's focus measure (Santos97) (MIS2). 
	 * http://www.sayonics.com/publications/pertuz_PR2013.pdf
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric_BREN (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize);

	/**
	 * @brief computeFocusMetric_CONT - This function measures the relative degree of focus of an image using the Image contrast (Nanda2001) (MIS3). 
	 * http://www.sayonics.com/publications/pertuz_PR2013.pdf
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric_CONT (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize);
	
    /**
	 * @brief computeFocusMetric_CURV - This function measures the relative degree of focus of an image using the Image Curvature (Helmli2001) (MIS4). 
	 * http://www.sayonics.com/publications/pertuz_PR2013.pdf
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric_CURV (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize);

    /**
	 * @brief computeFocusMetric_HELM - This function measures the relative degree of focus of an image using the Helmli's mean method (Helmli2001) (MIS5). 
	 * http://www.sayonics.com/publications/pertuz_PR2013.pdf
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric_HELM (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize);
	
	/**
	 * @brief computeFocusMetric_GDER - This function measures the relative degree of focus of an image using the Gaussian derivative (Geusebroek2000)        
	 * http://www.sayonics.com/publications/pertuz_PR2013.pdf
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric_GDER (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize);

	/**
	 * @brief computeFocusMetric_GLVA - This function measures the relative degree of focus of an image using the Graylevel variance (Krotkov86). 
	 * http://www.sayonics.com/publications/pertuz_PR2013.pdf
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric_GLVA (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize);

	/**
	 * @brief computeFocusMetric_GLLV - This function measures the relative degree of focus of an image using the Graylevel local variance (Pech2000)        
	 * http://www.sayonics.com/publications/pertuz_PR2013.pdf
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric_GLLV (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize);

	/**
	 * @brief computeFocusMetric_GLVN - This function measures the relative degree of focus of an image using the Normalized GLV (Santos97)
	 * http://www.sayonics.com/publications/pertuz_PR2013.pdf
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric_GLVN (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize);

	/**
	 * @brief computeFocusMetric_GRAE - This function measures the relative degree of focus of an image using the Energy of gradient (Subbarao92a)
	 * http://www.sayonics.com/publications/pertuz_PR2013.pdf
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric_GRAE (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize);
	
	/**
	 * @brief computeFocusMetric_HISR - This function measures the relative degree of focus of an image using the Histogram range (Firestone91)
	 * http://www.sayonics.com/publications/pertuz_PR2013.pdf
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric_HISR (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize);

	/**
	 * @brief computeFocusMetric_LAPE - This function measures the relative degree of focus of an image using the Energy of laplacian (Subbarao92a)
	 * http://www.sayonics.com/publications/pertuz_PR2013.pdf
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric_LAPE (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize);

	/**
	 * @brief computeFocusMetric_LAPM - This function measures the relative degree of focus of an image using the Variance of laplacian (Pech2000)
	 * http://www.sayonics.com/publications/pertuz_PR2013.pdf
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric_LAPM (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize);

	/**
	 * @brief computeFocusMetric_LAPV - This function measures the relative degree of focus of an image using the Modified Laplacian (Nayar89)
	 * http://www.sayonics.com/publications/pertuz_PR2013.pdf
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric_LAPV (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize);

	/**
	 * @brief computeFocusMetric_LAPD - This function measures the relative degree of focus of an image using the Diagonal laplacian (Thelen2009)
	 * http://www.sayonics.com/publications/pertuz_PR2013.pdf
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric_LAPD (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize);

	/**
	 * @brief computeFocusMetric_TENG - This function measures the relative degree of focus of an image using the Tenengrad (Krotkov86)
	 * http://www.sayonics.com/publications/pertuz_PR2013.pdf
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric_TENG (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize);

	/**
	 * @brief computeFocusMetric_TENV - This function measures the relative degree of focus of an image using the Tenengrad variance (Pech2000)
	 * http://www.sayonics.com/publications/pertuz_PR2013.pdf
	 * @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
	 * @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
	 * @param aWindowSize	The Window Size (only used on some operators)
	 **/
	static double computeFocusMetric_TENV (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize);

private: // Private classes

	/**
	 *@brief This class contains a list of the avaliable Focus Metrics function pointers
	 **/
	class FocusMetricLookup
	{
	public:
		FocusMetricLookup()
		{
			mFocusMetricLookup[FM_ACMO] = &computeFocusMetric_ACMO;
			mFocusMetricLookup[FM_BREN] = &computeFocusMetric_BREN;
			mFocusMetricLookup[FM_CONT] = &computeFocusMetric_CONT;
			mFocusMetricLookup[FM_CURV] = &computeFocusMetric_CURV;
			mFocusMetricLookup[FM_HELM] = &computeFocusMetric_HELM;
			mFocusMetricLookup[FM_GLVA] = &computeFocusMetric_GLVA;
			mFocusMetricLookup[FM_GLLV] = &computeFocusMetric_GLLV;
			mFocusMetricLookup[FM_GLVN] = &computeFocusMetric_GLVN;
			mFocusMetricLookup[FM_GRAE] = &computeFocusMetric_GRAE;
			mFocusMetricLookup[FM_HISR] = &computeFocusMetric_HISR;
			mFocusMetricLookup[FM_LAPE] = &computeFocusMetric_LAPE;
			mFocusMetricLookup[FM_LAPM] = &computeFocusMetric_LAPM;
			mFocusMetricLookup[FM_LAPV] = &computeFocusMetric_LAPV;
			mFocusMetricLookup[FM_LAPD] = &computeFocusMetric_LAPD;
			mFocusMetricLookup[FM_TENG] = &computeFocusMetric_TENG;
			mFocusMetricLookup[FM_TENV] = &computeFocusMetric_TENV;
		}
		FocusMap mFocusMetricLookup;
	};

private: // Note: Static Class only.... No instantiation... Remove Cononical Functions.
	FocusMetrics(void);										// Default Constructor
	~FocusMetrics(void);										// Default Destructor
	FocusMetrics& operator=(const FocusMetrics mRHS);		// Default operator =
	FocusMetrics(const FocusMetrics& mRHS);				// Default copy Constructor
};



/**
* @brief computeFocusMetric - This function measures the relative degree of focus of an image. 
* @param aImage   The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aHeight  The height of the image
* @param aWidth   The width of the image
* @param aMetric  The specific metric to compute based on.
* @param aROI     The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
**/
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric(const MatType &aSrc, const FOCUS_METRIC aMetric,const cv::Size aROI, const int32_t aWindowSize)
{
	// Only creaate lookup table once...
	static FocusMetricLookup LOOKUP;

	// Return result.
	double tFocusLevel = 0.0;

	// attempt to find the desired metric...
	FocusMap::const_iterator tIterFound = LOOKUP.mFocusMetricLookup.find(aMetric);
	FocusMap::const_iterator tIterEnd = LOOKUP.mFocusMetricLookup.end();

	// if the metric does exist then calcualte the value
	// else return a -1 for failure...
	if (tIterEnd != tIterFound)
	{
		std::vector<MatType> tPlanes(3);
		MatType tYUV;
		MatType tHist;
		cvtColor(aSrc, tYUV, cv::COLOR_BGR2YUV);
		split(aSrc,tPlanes);
		
		MatType image = tPlanes[0];
		FocusMetricPointer tFocusMetricPointer = tIterFound->second;
		tFocusLevel = (*tFocusMetricPointer)(image, aROI, aWindowSize);
	}

	// Return the function result.
	return tFocusLevel;
}

/**
* @brief computeFocusMetric_ACMO - This function measures the relative degree of focus of an image using the Absolute Central Moment (Shirvaikar2004) (MIS1). 
* http://www.sayonics.com/publications/pertuz_PR2013.pdf
* @param aImage		    The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
* [M N] = size(Image);
* Hist = imhist(Image)/(M*N);
* Hist = abs((0:255)-255*mean2(Image))'.*Hist;
* fm = sum(Hist);
**/

struct IncGenerator 
{
	float current_;
	IncGenerator (float start) : current_(start) {}
	float operator() () { return current_++; }
};

// FAIL????
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric_ACMO (const MatType & aSrc, const cv::Size, const int32_t)
{
	//function fm = AcMomentum(Image)
	//[M N] = size(Image);
	//Hist = imhist(Image)/(M*N);
	//Hist = abs((0:255)-255*mean2(Image))'.*Hist;
	//fm = sum(Hist);
	//end

	MatType tHistogram = calculateHistogrm(aSrc);
	multiply(1.0f/(aSrc.rows * aSrc.cols),tHistogram,tHistogram);

	std::vector<float> m(256);
	IncGenerator g (0);
	std::generate( m.begin(), m.end(), g); // Fill with the result of calling g() repeatedly.

	MatType M = MatType(256,1,CV_32F,&m[0]);

	Scalar mu, sigma;
    meanStdDev(aSrc, mu, sigma);

	M = M - (mu[0]*255);
	M = abs(M);

	multiply(M,tHistogram,M);

	Scalar fm = sum(M); 

	return fm[0];
}

/**
* @brief computeFocusMetric_BREN - This function measures the relative degree of focus of an image using the Brenner's focus measure (Santos97) (MIS2). 
* http://www.sayonics.com/publications/pertuz_PR2013.pdf
* @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aHeight		The height of the image
* @param aWidth		The width of the image
* @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
**/
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric_BREN (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize)
{
	//[M N] = size(Image);
	//DH = Image;
	//DV = Image;
	//DH(1:M-2,:) = diff(Image,2,1);
	//DV(:,1:N-2) = diff(Image,2,2);
	//FM = max(DH, DV);        
	//FM = FM.^2;
	//FM = mean2(FM);

	return 0;
}

/**
* @brief computeFocusMetric_CONT - This function measures the relative degree of focus of an image using the Image contrast (Nanda2001) (MIS3). 
* http://www.sayonics.com/publications/pertuz_PR2013.pdf
* @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aHeight		The height of the image
* @param aWidth		The width of the image
* @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
**/
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric_CONT (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize)
{
	//ImContrast = inline('sum(abs(x(:)-x(5)))');
	//FM = nlfilter(Image, [3 3], ImContrast);
	//FM = mean2(FM);

	return 0;
}

/**
* @brief computeFocusMetric_CURV - This function measures the relative degree of focus of an image using the Image Curvature (Helmli2001) (MIS4). 
* http://www.sayonics.com/publications/pertuz_PR2013.pdf
* @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aHeight		The height of the image
* @param aWidth		The width of the image
* @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
**/
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric_CURV (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize)
{
	//if ~isinteger(Image), Image = im2uint8(Image);
	//end
	//M1 = [-1 0 1;-1 0 1;-1 0 1];
	//M2 = [1 0 1;1 0 1;1 0 1];
	//P0 = imfilter(Image, M1, 'replicate', 'conv')/6;
	//P1 = imfilter(Image, M1', 'replicate', 'conv')/6;
	//P2 = 3*imfilter(Image, M2, 'replicate', 'conv')/10 ...
	//    -imfilter(Image, M2', 'replicate', 'conv')/5;
	//P3 = -imfilter(Image, M2, 'replicate', 'conv')/5 ...
	//    +3*imfilter(Image, M2, 'replicate', 'conv')/10;
	//FM = abs(P0) + abs(P1) + abs(P2) + abs(P3);
	//FM = mean2(FM);
	double m1[3][3] = {{-1,0,1},{-1,0,1},{-1,0,1}};
	double m2[3][3] = {{1,0,1},{1,0,1},{1,0,1}};
	Mat M1 = Mat(3, 3,CV_64F,m1);
	Mat M2 = Mat(3, 3,CV_64F,m2);
	Mat M1T;
	Mat M2T;

	transpose(M1,M1T);
	transpose(M2,M2T);

	MatType IM1;
	MatType IM1T;
	MatType IM2;
	MatType IM2T;
	filter2D(aSrc, IM1, -1 , M1);
	filter2D(aSrc, IM1T, -1 , M1T);
	filter2D(aSrc, IM2, -1 , M2);
	filter2D(aSrc, IM2T, -1 , M2T);

	//P0 = imfilter(Image, M1, 'replicate', 'conv')/6;
	MatType P0;
	multiply(1/6.0,IM1,P0);

	//P1 = imfilter(Image, M1', 'replicate', 'conv')/6;
	MatType P1;
	multiply(1/6.0,IM1T,P1);

	//P2 = 3*imfilter(Image, M2, 'replicate', 'conv')/10 - imfilter(Image, M2', 'replicate', 'conv')/5;
	MatType P2A;
	MatType P2B;
	MatType P2;
	multiply(3.0/10.0,IM2,P2A);
	multiply(1.0/5.0,IM2T,P2B);
	subtract(P2A,P2B,P2);
	
	//P3 = -imfilter(Image, M2, 'replicate', 'conv')/5 + 3*imfilter(Image, M2, 'replicate', 'conv')/10;	
	MatType P3A;
	MatType P3B;
	MatType P3;
	multiply(-1.0/5.0,IM2,P3A);
	multiply(3.0/10.0,IM2,P3B);
	add(P3A,P3B,P3);

	//FM = abs(P0) + abs(P1) + abs(P2) + abs(P3);
	MatType FM = abs(P0) + abs(P1) + abs(P2) + abs(P3);
	
	//FM = mean2(FM);
	Scalar tMean;
	Scalar tStddev;
	meanStdDev(FM,tMean,tStddev);
	//return tMean[0];
	return 0;
}

/**
* @brief computeFocusMetric_HELM - This function measures the relative degree of focus of an image using the Helmli's mean method (Helmli2001) (MIS5). 
* http://www.sayonics.com/publications/pertuz_PR2013.pdf
* @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aHeight		The height of the image
* @param aWidth		The width of the image
* @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
**/
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric_HELM (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize)
{
	/*
	//MEANF = fspecial('average',[WSize WSize]);
	float MEANF[aWindowSize][aWindowSize];
	for (int ii = 0 ; ii < aWindowSize; ii++)
	{
		for (int ii = 0 ; ii < aWindowSize; ii++)
		{
			MEANF[ii][jj] = 1/(aWindowSize*aWindowSize);
		}
	}

	//U = imfilter(Image, MEANF, 'replicate');
	Point anchor( 0 ,0 );
	double delta = 0;

	MatType ker = MatType(aWindowSize, aWindowSize, CV_32FC1, &MEANF);
	MatType U = MatType(aSrc.size(), aSrc.type());

	Ptr<FilterEngine> fe =  createLinearFilter(aSrc.type(), ker.type(), ker, anchor, delta, BORDER_CONSTANT, BORDER_CONSTANT, Scalar(0));
	fe->apply(aSrc, dst);
	
	//R1 = U./Image;
	MatType R1;
	cv::divide(U,aSrc,R1);

	//R1(Image==0)=1;
	//index = (U>Image);
	//FM = 1./R1;
	//FM(index) = R1(index);
	//FM = mean2(FM);
	*/
	return 0;
}

/**
* @brief computeFocusMetric_HELM - This function measures the relative degree of focus of an image using the Graylevel variance (Krotkov86). 
* http://www.sayonics.com/publications/pertuz_PR2013.pdf
* @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
* DONE!!!
**/
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric_GLVA (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize)
{
	// MATLAB
	//case 'GLVA' % Graylevel variance (Krotkov86)
	//FM = std2(Image);
	Scalar tMean;
	Scalar tStddev;
	meanStdDev(aSrc,tMean,tStddev);
	return tStddev[0];
}
/**
* @brief computeFocusMetric_GLLV - This function measures the relative degree of focus of an image using the %Graylevel local variance (Pech2000). 
* http://www.sayonics.com/publications/pertuz_PR2013.pdf
* @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
**/
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric_GLLV (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize)
{
	//MATLAB
	//case 'GLLV' %Graylevel local variance (Pech2000)        
    //LVar = stdfilt(Image, ones(WSize,WSize)).^2;
    //FM = std2(LVar)^2;

	/*
	Stdfilt in Opencv
import cv2
import numpy as np
 
img = cv2.imread('fruits.jpg', True)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = img / 255.0
 
# c = imfilter(I,h,'symmetric');
h = np.ones((3,3))
n = h.sum()
n1 = n - 1
c1 = cv2.filter2D(img**2, -1, h/n1, borderType=cv2.BORDER_REFLECT)
c2 = cv2.filter2D(img, -1, h, borderType=cv2.BORDER_REFLECT)**2 / (n*n1)
J = np.sqrt( np.maximum(c1-c2,0) )
 
cv2.imshow('stdfilt', J)
cv2.waitKey(0)
cv2.destroyWindow('stdfilt')
       
I = imread('fruits.jpg');
I = im2double(rgb2gray(I));
imshow(stdfilt(I))
	*/
	return 0;
}
/**
* @brief computeFocusMetric_GLVN - This function measures the relative degree of focus of an image using the Normalized GLV (Santos97)
* http://www.sayonics.com/publications/pertuz_PR2013.pdf
* @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
* DONE!!!
**/
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric_GLVN (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize)
{
	// MATLAB
	//case 'GLVN' % Normalized GLV (Santos97)
    //FM = std2(Image)^2/mean2(Image);
	Scalar mu, sigma;
    meanStdDev(aSrc, mu, sigma);
    return (sigma.val[0]*sigma.val[0]) / mu.val[0];;
}
/**
* @brief computeFocusMetric_GRAE - This function measures the relative degree of focus of an image using the Energy of gradient (Subbarao92a)
* http://www.sayonics.com/publications/pertuz_PR2013.pdf
* @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
**/
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric_GRAE (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize)
{
	//case 'GRAE' % Energy of gradient (Subbarao92a)
	//Ix = Image;
	//Iy = Image;
	//Iy(1:end-1,:) = diff(Image, 1, 1);
	//Ix(:,1:end-1) = diff(Image, 1, 2);
	//FM = Ix.^2 + Iy.^2;
	//FM = mean2(FM);
	return 0;
}

/**
* @brief computeFocusMetric_LAPE - This function measures the relative degree of focus of an image using the Energy of laplacian (Subbarao92a)
* http://www.sayonics.com/publications/pertuz_PR2013.pdf
* @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
**/
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric_LAPE (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize)
{
	//MATLAB
	//case 'LAPE' % Energy of laplacian (Subbarao92a)
	//LAP = fspecial('laplacian');
	//FM = imfilter(Image, LAP, 'replicate', 'conv');
	//FM = mean2(FM.^2);
	MatType FM;
	ImageFilters<MatType>::process(aSrc, FM, FILTER_LAPLACIAN);
	MatType FM2;
	multiply(FM,FM,FM2);
	return mean(FM2).val[0];
}

/**
* @brief computeFocusMetric_LAPM - This function measures the relative degree of focus of an image using the Modified Laplacian (Nayar89)
* http://www.sayonics.com/publications/pertuz_PR2013.pdf
* @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
**/
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric_LAPM (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize)
{
	// MATLAB    
	//case 'LAPM' % Modified Laplacian (Nayar89)
	//M = [-1 2 -1];        
	//Lx = imfilter(Image, M, 'replicate', 'conv');
	//Ly = imfilter(Image, M', 'replicate', 'conv');
	//FM = abs(Lx) + abs(Ly);
	//FM = mean2(FM);

	Mat M = (Mat_<double>(3, 1) << -1, 2, -1);
	Mat G = getGaussianKernel(3, -1, CV_64F);

	MatType Lx;
    sepFilter2D(aSrc, Lx, CV_64F, M, G);

	MatType Ly;
    sepFilter2D(aSrc, Ly, CV_64F, G, M);

	MatType FM;
	add(abs(Lx),abs(Ly),FM);

    return mean(FM).val[0];
}

/**
* @brief computeFocusMetric_LAPV - This function measures the relative degree of focus of an image using the Modified Laplacian (Nayar89)
* http://www.sayonics.com/publications/pertuz_PR2013.pdf
* @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
**/
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric_LAPV (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize)
{
	//case 'LAPV' % Variance of laplacian (Pech2000)
	//LAP = fspecial('laplacian');
	//ILAP = imfilter(Image, LAP, 'replicate', 'conv');
	//FM = std2(ILAP)^2;
	MatType FM;
	Laplacian(aSrc, FM, CV_64F);
	Scalar tMean;
	Scalar tStddev;
	meanStdDev(FM,tMean,tStddev);
	return tStddev[0]*tStddev[0];
}
/**
* @brief computeFocusMetric_LAPD - This function measures the relative degree of focus of an image using the Diagonal laplacian (Thelen2009)
* http://www.sayonics.com/publications/pertuz_PR2013.pdf
* @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
**/
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric_LAPD (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize)
{
	//case 'LAPD' % Diagonal laplacian (Thelen2009)
	//M1 = [-1 2 -1];
	//M2 = [0 0 -1;0 2 0;-1 0 0]/sqrt(2);
	//M3 = [-1 0 0;0 2 0;0 0 -1]/sqrt(2);
	//F1 = imfilter(Image, M1, 'replicate', 'conv');
	//F2 = imfilter(Image, M2, 'replicate', 'conv');
	//F3 = imfilter(Image, M3, 'replicate', 'conv');
	//F4 = imfilter(Image, M1', 'replicate', 'conv');
	//FM = abs(F1) + abs(F2) + abs(F3) + abs(F4);
	//FM = mean2(FM);
	double m1[3] = {-1,2,-1};
	double m2[3][3] = {{0,0,-1},{0,2,0},{-1,0,0}};
	double m3[3][3] = {{-1,0,0},{0,2,0},{0,0,-1}};
	Mat M1 = Mat(3, 1,CV_64F,m1);
	Mat M2 = Mat(3, 3,CV_64F,m2);
	Mat M3 = Mat(3, 3,CV_64F,m3);
	Mat G = getGaussianKernel(3, -1, CV_64F);

	MatType F2;
	MatType F3;
    filter2D(aSrc, F2, -1 , M2);
	filter2D(aSrc, F3, -1 , M3);
	add(abs(F3),abs(F2),F3);
	
	MatType F4;
    MatType F1;
    sepFilter2D(aSrc, F1, CV_64F, M1, G);
	sepFilter2D(aSrc, F4, CV_64F, G, M1);
	add(abs(F1),abs(F4),F1);

	F1.convertTo(F1,F2.type());
	add(F3,F1,F1);
    return mean(F1).val[0];

}

/**
* @brief computeFocusMetric_TENG - This function measures the relative degree of focus of an image using the Tenengrad (Krotkov86)
* http://www.sayonics.com/publications/pertuz_PR2013.pdf
* @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
**/
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric_TENG (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize)
{
	//case 'TENG'% Tenengrad (Krotkov86)
	//Sx = fspecial('sobel');
	//Gx = imfilter(double(Image), Sx, 'replicate', 'conv');
	//Gy = imfilter(double(Image), Sx', 'replicate', 'conv');
	//FM = Gx.^2 + Gy.^2;
	//FM = mean2(FM);
    Mat Gx, Gy;
    Sobel(aSrc, Gx, CV_64F, 1, 0, 3);
    Sobel(aSrc, Gy, CV_64F, 0, 1, 3);
	multiply(Gx,Gx,Gx);
	multiply(Gy,Gy,Gy);
    MatType FM;
	add(Gx,Gy,FM);
    return mean(FM).val[0];
}
/**
* @brief computeFocusMetric_TENV - This function measures the relative degree of focus of an image using the Tenengrad variance (Pech2000)
* http://www.sayonics.com/publications/pertuz_PR2013.pdf
* @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
**/
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric_TENV (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize)
{

   // case 'TENV' % Tenengrad variance (Pech2000)
    //    Sx = fspecial('sobel');
    //    Gx = imfilter(double(Image), Sx, 'replicate', 'conv');
    //    Gy = imfilter(double(Image), Sx', 'replicate', 'conv');
    //    G = Gx.^2 + Gy.^2;
    //    FM = std2(G)^2;
    MatType Gx, Gy;
    Sobel(aSrc, Gx, CV_64F, 1, 0, 3);
    Sobel(aSrc, Gy, CV_64F, 0, 1, 3);
	multiply(Gx,Gx,Gx);
	multiply(Gy,Gy,Gy);
    MatType FM;
	add(Gx,Gy,FM);
	Scalar tStddev;
	Scalar tMean;
	meanStdDev(FM,tMean,tStddev);
	return tStddev[0]*tStddev[0];
}
/**
* @brief computeFocusMetric_HISR - This function measures the relative degree of focus of an image using the Histogram range (Firestone91)
* http://www.sayonics.com/publications/pertuz_PR2013.pdf
* @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
**/
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric_HISR (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize)
{
	//case 'HISR' % Histogram range (Firestone91)
    //    FM = max(Image(:))-min(Image(:));
	double minVal, maxVal;
	minMaxLoc(aSrc, &minVal, &maxVal); //find minimum and maximum intensities
	return maxVal-minVal;   
}

/**
* @brief computeFocusMetric_GDER - This function measures the relative degree of focus of an image using the Gaussian derivative (Geusebroek2000)        
* http://www.sayonics.com/publications/pertuz_PR2013.pdf
* @param aImage		The image to compute the focus level on - The Image is assumed to be a grayscale image.
* @param aROI			The ROI of target region to compute focus metric on ( [X_0 Y_0 Width Height]) ... Note: This is optional.
* @param aWindowSize	The Window Size (only used on some operators)
**/
template<typename MatType>
double FocusMetrics<MatType>::computeFocusMetric_GDER (const MatType & aSrc,const cv::Size aROI, const int32_t aWindowSize)
{/*
    case 'GDER' % Gaussian derivative (Geusebroek2000)        
        N = floor(WSize/2);
        sig = N/2.5;
        [x,y] = meshgrid(-N:N, -N:N);
        G = exp(-(x.^2+y.^2)/(2*sig^2))/(2*pi*sig);
        Gx = -x.*G/(sig^2);Gx = Gx/sum(Gx(:));
        Gy = -y.*G/(sig^2);Gy = Gy/sum(Gy(:));
        Rx = imfilter(double(Image), Gx, 'conv', 'replicate');
        Ry = imfilter(double(Image), Gy, 'conv', 'replicate');
        FM = Rx.^2+Ry.^2;
        FM = mean2(FM);
        
		*/
	//int N = 
	//Scalar tStddev;
	//Scalar tMean;
	//meanStdDev(FM,tMean,tStddev);
	return 0;
}
