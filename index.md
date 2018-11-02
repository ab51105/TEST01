# 黃世耀 <span style="color:red">(106061561)</span>

# Project 2 / Panorama Stitching

## Overview
The project is related to 
> image stitching
 


## Implementation
1. Get SIFT points and descriptors
	
	In this part, we install cyvlfeat in Anaconda. And use the built-in function "sift"  to fetch sift features in all input images.
2. Matching SIFT Descriptors

	We need to match descriptors together in two images. So in this part, we need to find one match descriptor for each features in descriptor1. The code below is how I implement `SIFTSimpleMatcher`. At first, I will copy each row of descriptor1 'descriptor2.shape[0]' times and copy the whole descriptor2 numpy array for 'descriptor1.shape[0]' times. With this operation, we can calculate the distance of all descriptors between two images' sift features by matrix operation easily and fastly. After finding the distances between sift features, we can find the minimum distances for each descriptor1 and if its value is less than `second smallest distances * THRESH(=0.7 here)`, we call the two descriptors match.
	```
	def SIFTSimpleMatcher(descriptor1, descriptor2, THRESH=0.7):
		# size
		descriptor1_len = descriptor1.shape[0];
		descriptor2_len = descriptor2.shape[0];
		
		# modify array for matrix operation
		descriptor1_cpy = np.repeat(descriptor1,descriptor2_len,axis=0);
		descriptor2_cpy = np.tile(descriptor2,(descriptor1_len,1));
		
		# calculate distance
		descriptor_dist = descriptor1_cpy - descriptor2_cpy;
		descriptor_dist = np.square(descriptor_dist);
		descriptor_dist_sum = np.sum(descriptor_dist,axis=1);
		descriptor_dist_sum = np.sqrt(descriptor_dist_sum);
		
		# match
		match = np.array([[0,0]]);
		match_num = 0;
		for i in range(descriptor1_len):
			min_dist = np.min(descriptor_dist_sum[i*descriptor2_len : (i+1)*descriptor2_len]);
			min_idx = np.argmin(descriptor_dist_sum[i*descriptor2_len : (i+1)*descriptor2_len]);
			descriptor_dist_sum[i*descriptor2_len + min_idx] = 1000000;
			second_min_dist = np.min(descriptor_dist_sum[i*descriptor2_len : (i+1)*descriptor2_len]);
			if min_dist <= second_min_dist * THRESH:
				if match_num == 0:
					match[match_num,0] = i;
					match[match_num,1] = min_idx;
					match_num = match_num + 1;
				else:
					match = np.append(match,[[i,min_idx]],axis=0);
		return match
	``` 
3. Fitting the Transformation Matrix

	In this part, we calculate the transformation matrix. The transformation matrix here is to map the feature points in image1 to feature points in image2. So we only need to modify origin transformation formula and solve the linear equation below:
	```
	   P2 = H * P1
	=>(P2)' = (H * P1)'
	=> P1' * H' = P2'
	   (Ax = B) 
	```
	So, in function `ComputeAffineMatrix`, the H can be calculate with `np.linalg.lstsq`:
	```
	H_transpose = np.linalg.lstsq(np.transpose(P1),np.transpose(P2))[0];
    H = np.transpose(H_transpose);
	H[2,:] = [0, 0, 1]
	```
4. RANSAC

	In the previous step, we can find Homography between two images, but it is not the accurate homography. In order to get a better one, we use RANSAC("RANdom SAmple Consensus") method to find a better Homography. It iterates several times, each time it random select some match points out. And in these match points, we only uses “inliers” to compute the Homography and use this homogrphy to calculate its cost function. After all the iterations, we choose the best homography with the lowest cost. The `ComputeError` code is showed below, the only thing we need to do is to map one image to another image and compute the distance between mapping point and corresponding feature point. And then we can get the error.
	```
	def ComputeError(H, pt1, pt2, match):
		pt1_addone = np.concatenate([pt1,np.ones([np.shape(pt1)[0],1])],axis = 1);
		pt1_trans_to_pt2 = np.dot(H,pt1_addone[match[:, 0],:].T);
		pt2_addone = np.concatenate([pt2,np.ones([np.shape(pt2)[0],1])],axis = 1);
		dist_sub = pt1_trans_to_pt2 - pt2_addone[match[:,1],:].T;
		dists = np.sqrt(np.sum((np.square(dist_sub).T),axis = 1));

		if len(dists) != len(match):
        	sys.exit('wrong format')
    	return dists
	```
5. Stitching ordered sequence of images

	The images in this part are ordered from left to right. And then we will choose the middle image as reference and calculate the corresponding homography to middle image(reference). The new homography is calculated by the rule below:
	```
	case 1: refFrameIndex > currentFrameIndex
		H(new) = H(ref-1) * H(ref-2) * H(now+1) *...* H(now) 
	case 2: refFrameIndex < currentFrameIndex
		H(new) = inv[H(ref)] * inv[H(ref+1)] *...* inv[H(now-2)] * inv[H(now-1)]
	case 3: refFrameIndex = currentFrameIndex
		H(new) = identity matrix
	```
	And the code is showed below:
	```
	def makeTransformToReferenceFrame(i_To_iPlusOne_Transform, currentFrameIndex, refFrameIndex):
		if refFrameIndex > currentFrameIndex:
			T = i_To_iPlusOne_Transform[refFrameIndex-1];
			for i in range(refFrameIndex-2,currentFrameIndex-1,-1):
				T = np.dot(T,i_To_iPlusOne_Transform[i]);
		elif refFrameIndex == currentFrameIndex:
			T = np.eye(3);
		else:
			T = np.linalg.pinv(i_To_iPlusOne_Transform[refFrameIndex]);
			for i in range(refFrameIndex+1,currentFrameIndex):
				T = np.dot(T,np.linalg.pinv(i_To_iPlusOne_Transform[i]));
		return T 
	``` 
	After calculating the new homography, we can stitch all the image together and get the Panorama Stitching image.  


## Installation
* Other required packages.

	cyvlfeat for fetching sift features



## Results

<table border=1>
<tr>
	<td>
	Hanging_pano.png
	</td>
	<td>
	<img src="Hanging_pano.png"/>
	</td>
</tr>

<tr>
	<td>
	MelakwaLake_pano.png
	</td>
	<td>
	<img src="MelakwaLake_pano.png"/>
	</td>
</tr>

<tr>
	<td>
	Rainier_pano.png
	</td>
	<td>
	<img src="Rainier_pano.png"/>
	</td>
</tr>

<tr>
	<td>
	uttower_pano.png
	</td>
	<td>
	<img src="uttower_pano.png"/>
	</td>
</tr>

<tr>
	<td>
	yosemite_pano.png
	</td>
	<td>
	<img src="yosemite_pano.png"/>
	</td>
</tr>
</table>

## My image
<table>
<tr>
	<td>
	images
	</td>
	<td>
	<img src="../data/park1.png"/>
	</td>
	<td>
	<img src="../data/park2.png"/>
	</td>
	<td>
	<img src="../data/park3.png"/>
	</td>
</tr>
</table>
Stitching images
<img src="park_pano.png"/>

