%> ------------------------------------------------------------------------
%> Write image file names and their corresponding GT Pose
%> ------------------------------------------------------------------------
clc; clear all; close all;

%> Read all images in the sequence.
%> Use imread(Image_Sequence(i).name); to read image i
mfiledir = fileparts(mfilename('fullpath'));
Image_Sequence = dir([mfiledir,'/MyData/Problem2/fr2_desk_tmp/*.png']);
Img_GT_Association = importdata([mfiledir, '/MyData/Problem2/associate_img_gt.txt']);
Img_Names = Img_GT_Association.textdata;
GT_Poses = Img_GT_Association.data;

%> Write to Files
Output_Img_File_Path = [mfiledir, '/Image_List_.txt'];
Output_Img_File_Write = fopen(Output_Img_File_Path, 'w');
Output_GT_File_Path = [mfiledir, '/GT_Pose_List_.txt'];
Output_GT_File_Write = fopen(Output_GT_File_Path, 'w');

start_j = 1;
for i = 1:size(Image_Sequence, 1)
    Img_Name = strcat("rgb/", Image_Sequence(i).name);
    for j = start_j:size(Img_Names, 1)
        Assoc_Name = string(Img_Names{j,2});
        if strcmp(Assoc_Name, Img_Name)
            fprintf(Output_Img_File_Write, Image_Sequence(i).name);
            fprintf(Output_Img_File_Write, "\n");
            for gi = 2:8
                fprintf(Output_GT_File_Write, string(GT_Poses(j,gi))); 
                fprintf(Output_GT_File_Write, "\t"); 
            end
            fprintf(Output_GT_File_Write, "\n");
            start_j = j;
            break;
        end
    end
end

fclose(Output_Img_File_Write);
fclose(Output_GT_File_Write);

%% From fr2_desl_tmp to fr2_desk: imges with associated GT
% 
% Source_Img_Path      = strcat(mfiledir,'/MyData/Problem2/fr2_desk_tmp/');
% Destination_Img_Path = strcat(mfiledir,'/MyData/Problem2/fr2_desk/');
% Image_List_Data      = string(importdata([mfiledir, '/Image_List.txt']));
% % GT_List_Data = importdata([mfiledir, '/GT_Pose_List.txt']);
% 
% for k = 1:size(Image_List_Data,1)
%     Source_File = strcat(Source_Img_Path, Image_List_Data(k,1));
%     Dest_File   = fullfile(Destination_Img_Path, Image_List_Data(k,1));
%     movefile(Source_File, Dest_File);
% end

