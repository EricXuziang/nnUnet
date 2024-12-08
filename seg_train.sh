# for fold in {0..5}
# do 
#     # echo "nnUNetv2_train 1 3d_lowres $fold"
#     nnUNetv2_train 001 3d_fullres $fold -p nnUNetResEncUNetMPlans
# done

nnUNetv2_train 001 3d_fullres 0 -p nnUNetResEncUNetMPlans

# nnUNetv2_predict -i /well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_raw/Dataset001_quiz/imagesVal -o /well/rittscher/users/ycr745/nnUNet/nnunetv2/results/val -c 3d_fullres -d 001 -f 0