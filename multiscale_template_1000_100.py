# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2
import time
import os
import pandas as pd
from remove_black import removeBlackRegion,crop_points_for_black_color

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=False, help="Path to template image")
ap.add_argument("--image", default=None,
	help="Path to image where template will be matched")
ap.add_argument("--image_folder",default=None,
	help="Path to image folder where template will be matched")
ap.add_argument("--root_folder",default=None,
	help="Path to image folder where template will be matched")
ap.add_argument("--num_images",default=None,
	help="Path to image folder where template will be matched")
ap.add_argument("--template_folder",default=None,
	help="Path to image folder where template will be matched")
ap.add_argument("-v", "--visualize",
	help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())
# load the image image, convert it to grayscale, and detect edges


def match_images(template,image):

    (tH, tW) = template.shape[:2]
    # image = removeBlackRegion(image)
    # row_limit_top, col_limit_top, row_limit_lower, col_limit_lower = crop_points_for_black_color(image)
    # image = image[row_limit_top:row_limit_lower,col_limit_top:col_limit_lower]

    # image = cv2.resize(image, (3264,2448))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edged_image = cv2.Canny(gray, 20, 60)
    # cv2.imwrite("edfeee.jpg",edged_image)

    # gray = cv2.resize(gray, (3264,2448))

    found = None


    # loop over the scales of the image
    for scale in np.linspace(0.1,1, 10)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        # scale = 2
        resized = imutils.resize(template, width = int(template.shape[1] * scale))
        r = template.shape[1] / float(resized.shape[1])

        # print (resized.shape)
        # if the resized image is smaller than the template, then break
        # from the loop
        if edged_image.shape[0] < tH or edged_image.shape[1] < tW:
            break

            
        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged_template = cv2.Canny(resized, 20, 60)


        # cv2.imwrite("edge_100x.jpg",edged)

        result = cv2.matchTemplate(edged_image, edged_template, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            # print (edged_template.shape)
            # print (maxVal)

            if (edged_template.shape[0] < edged_image.shape[0]/20)  or (edged_template.shape[1] < edged_image.shape[1]/20):
                pass
            else:
                # print (maxVal)
                found = (maxVal, maxLoc, r,edged_template.shape[0],edged_template.shape[1])
            # if edged_template.shape[0] == 513:
                # break
            

    
    return found



def match(template,image_path=None,image_folder=None,root_folder=None,num_images_folder=None,template_name= None):
    (tH, tW) = template.shape[:2]

    if image_folder is not None:
        folder_name = root_folder+"/"+template_name+"_outputs" 
        if not(os.path.exists(folder_name)):
            os.mkdir(folder_name)

        all_images = []
        all_scores = []
        maxVal_global = -1

        for i in range(1,num_images_folder):

            if (i%3 != 0):
                continue
            image_name = str(i)
            imagePath = image_folder+"/"+image_name+".jpg"
            # load the image, convert it to grayscale, and initialize the
            # bookkeeping variable to keep track of the matched region
            print (i)
            image = cv2.imread(imagePath)

            found = match_images(template,image)

            # (_, maxLoc, r) = found
            (_, maxLoc, r,tH_updated,tW_updated) = found

            (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
            # print (startX,startY)
            (endX, endY) = (int((maxLoc[0] + tW_updated)), int((maxLoc[1] + tH_updated)))
            # print (endX,endY)
            # found_img = image[startY:endY,startX:endX]

            if (found[0] > maxVal_global):
                winner_image = image_name
                maxVal_global = found[0]
                found_img = image[startY:endY,startX:endX]
                all_images.append(winner_image)
                all_scores.append(maxVal_global)
                print (winner_image)
                print (maxVal_global)

            # draw a bounding box around the detected result and display the image
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            # cv2.imshow("Image", image)
            # cv2.waitKey(0)
            # name_out = root_folder+"/outputs/output_"+image_name+".jpg"
            # # print (name_out)
            # cv2.imwrite(name_out,image)

            name_out = root_folder+"/"+template_name+"_outputs/output_"+image_name+".jpg"
            # print (name_out)
            cv2.imwrite(name_out,image)


        return all_images,all_scores,found_img,winner_image,maxVal_global
    
    elif image_path is not None:
        image = cv2.imread(image_path)
        image_dir = image_path.split(".")[0]

        found = match_images(template,image)

        (_, maxLoc, r,tH_updated,tW_updated) = found
        print (tH_updated,tW_updated)
        print (r)

        (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
        print (startX,startY)
        (endX, endY) = (int((maxLoc[0] + tW_updated)), int((maxLoc[1] + tH_updated)))
        print (endX,endY)
        found_img = image[startY:endY,startX:endX]


        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        print (found[0])
        name_out = image_dir+"_output.jpg"
        cv2.imwrite(name_out,image)
        return found_img
 



# template = removeBlackRegion(template)
# row_limit_top, col_limit_top, row_limit_lower, col_limit_lower = crop_points_for_black_color(template)
# template = template[row_limit_top:row_limit_lower,col_limit_top:col_limit_lower]


# cv2.imwrite(template_path+"_output_crop.jpg",template)

# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# template = cv2.resize(template,(300,300))
# template = cv2.Canny(template, 50, 100)

# print ("edge unique values    ",np.unique(template))


# cv2.imwrite(template_path+"_output_edge.jpg",template)

if args["image"] is not None:
    time_start = time.time()
    template_path = args["template"].split(".")[0]
    template = cv2.imread(args["template"])
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(template_path+"_output_edge.jpg",template)


    found_image = match(template,image_path=args["image"])
    cv2.imwrite(args["image"].split(".")[0]+"_found_template_rectangular.jpg",found_image)
elif args["image_folder"] is not None:

    time_start = time.time()


    for template_path in glob.glob(args["template_folder"]+"/*template_*.*"):
        template_name = template_path.split("/")[-1].split(".")[0]
        template = cv2.imread(template_path)

        print (template_name)


        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)


        cv2.imwrite(args["template_folder"]+"/"+template_name+"_output_edge.jpg",template)


        all_images,all_scores,found_img,winner_image,maxVal_global = match(template,image_folder=args["image_folder"],root_folder=args["root_folder"],num_images_folder=int(args["num_images"]),template_name=template_name)


        df = pd.DataFrame({"Image Name":all_images,"Scores":all_scores})

        df.to_csv(args["root_folder"]+"/stats.csv")
        df.to_csv(args["root_folder"]+"/"+template_name+"_stats.csv")


        print (winner_image)
        print (maxVal_global)
        cv2.imwrite(args["root_folder"]+"/found_template_rectangular.jpg",found_img)
        cv2.imwrite(args["root_folder"]+"/"+template_name+"_found_template_rectangular.jpg",found_img)
        print ("Total time: ",time.time() - time_start)



    # all_images,all_scores,found_img,winner_image,maxVal_global = match(template,image_folder=args["image_folder"],root_folder=args["root_folder"])


    # df = pd.DataFrame({"Image Name":all_images,"Scores":all_scores})

    # df.to_csv(args["root_folder"]+"/stats.csv")

    # print (winner_image)
    # print (maxVal_global)
    # cv2.imwrite(args["root_folder"]+"/found_template_rectangular.jpg",found_img)
    # print ("Total time: ",time.time() - time_start)
