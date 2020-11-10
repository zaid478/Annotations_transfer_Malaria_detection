# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2
import os
import time
import pandas as pd

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


def match_images(template,image):

    (tH, tW) = template.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found = None


    # loop over the scales of the image
    for scale in np.linspace(0.4,1, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

            
        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 20, 50)



        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    
    return found





def match(template,image_path=None,image_folder=None,root_folder=None,num_images_folder=None,template_name=None):
    (tH, tW) = template.shape[:2]

    if image_folder is not None:
        os.mkdir(root_folder+"/"+template_name+"_outputs")

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

            (_, maxLoc, r) = found


            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + tW)*r), int((maxLoc[1] + tH)*r))



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
            name_out = root_folder+"/"+template_name+"_outputs/output_"+image_name+".jpg"
            cv2.imwrite(name_out,image)

        return all_images,all_scores,found_img,winner_image,maxVal_global
    
    elif image_path is not None:
        image = cv2.imread(image_path)
        image_dir = image_path.split(".")[0]

        found = match_images(template,image)

        (_, maxLoc, r) = found

        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))


        found_img = image[startY:endY,startX:endX]

        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        print (found[0])
        name_out = image_dir+"_output.jpg"
        cv2.imwrite(name_out,image)
        return found_img
 


if args["image"] is not None:
    found_image = match(template,image_path=args["image"])
    cv2.imwrite(args["image"].split(".")[0]+"_found_template_rectangular.jpg",found_image)
elif args["image_folder"] is not None:

    time_start = time.time()


    for template_path in glob.glob(args["template_folder"]+"/*template_*.*"):
        template_name = template_path.split("/")[-1].split(".")[0]
        template = cv2.imread(template_path)

        print (template_name)


        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.resize(template,(300,300))
        template = cv2.Canny(template, 20, 50)


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
