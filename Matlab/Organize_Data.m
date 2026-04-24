%%
load('Abs_Poses.mat');

%> Write the estimations to a file
mfiledir = fileparts(mfilename('fullpath'));
Image_Sequence = dir([mfiledir, '/MyData/Problem2/fr2_desk/*.png']);

Output_Estimation_w_Time_Path = [mfiledir, '/Estimations.txt'];
Output_Estimation_w_Time_Write = fopen(Output_Estimation_w_Time_Path, 'w');
Output_GT_w_Time_Path = [mfiledir, '/GroundTruths.txt'];
Output_GT_w_Time_Write = fopen(Output_GT_w_Time_Path, 'w');
GT_Poses = importdata([mfiledir, '/GT_Pose_List.txt']);
for ci = 1:size(Image_Sequence, 1)
    TimeStamp = extractBefore(Image_Sequence(ci).name, ".png");
    R = Abs_Poses(:,1:3,ci);
    Q = rotm2quat(R);
    T = Abs_Poses(:,4,ci);
    C = -R'*T;
    C_str = strcat(string(C(1)), " ", string(C(2)), " ", string(C(3)));
    Q_str = strcat(string(Q(4)), " ", string(Q(1)), " ", string(Q(2)), " ", string(Q(3)));
    fprintf(Output_Estimation_w_Time_Write, strcat(TimeStamp, " ", C_str, " ", Q_str, "\n"));

    GT_str = strcat(string(GT_Poses(ci,1)), " ", string(GT_Poses(ci,2)), " ", string(GT_Poses(ci,3)), " ", ...
                    string(GT_Poses(ci,4)), " ", string(GT_Poses(ci,5)), " ", string(GT_Poses(ci,6)), " ", string(GT_Poses(ci,7)), "\n");
    fprintf(Output_GT_w_Time_Write, strcat(TimeStamp, " ", GT_str));
end

fclose(Output_Estimation_w_Time_Write);
fclose(Output_GT_w_Time_Write);

%% > Transform the trajectory by calculating relative poses

GT_Poses = importdata([mfiledir, '/GT_Pose_List.txt']);
Prev_Abs_R = quat2rotm([GT_Poses(1,7), GT_Poses(1,4), GT_Poses(1,5), GT_Poses(1,6)]);
Prev_Abs_C = [GT_Poses(1,1); GT_Poses(1,2); GT_Poses(1,3)];
Prev_Abs_T = -Prev_Abs_R * Prev_Abs_C;

% Output_Reorganized_GT_Path = [mfiledir, '/Reorganized_GT.txt'];
% Output_Reorganized_GT_Write = fopen(Output_Reorganized_GT_Path, 'w');

Prev_Abs_R_ = eye(3);
Prev_Abs_T_ = [0; 0; 0];

Abs_Poses_GT = zeros(3,4,size(GT_Poses,1));
Abs_Cam_Center_GT = zeros(3,size(GT_Poses,1));

for mi = 1:size(GT_Poses, 1)

    Curr_Abs_R = quat2rotm([GT_Poses(mi,7), GT_Poses(mi,4), GT_Poses(mi,5), GT_Poses(mi,6)]);
    Curr_Abs_C = [GT_Poses(mi,1); GT_Poses(mi,2); GT_Poses(mi,3)];
    % Curr_Abs_T = -Curr_Abs_R * Curr_Abs_C;

    Rel_R = Curr_Abs_R' * Prev_Abs_R;
    Rel_T = Curr_Abs_R' * (Prev_Abs_C - Curr_Abs_C);

    Curr_Abs_R_ = Rel_R * Prev_Abs_R_;
    Curr_Abs_T_ = Rel_R * Prev_Abs_T_ + Rel_T;
    Abs_Poses_GT(:,:,mi) = [Curr_Abs_R_, Curr_Abs_T_];
    Abs_Cam_Center_GT(:,mi) = -Curr_Abs_R_' * Curr_Abs_T_;

    % fprintf(Output_Reorganized_GT_Write);

    Prev_Abs_R_ = Curr_Abs_R_;
    Prev_Abs_T_ = Curr_Abs_T_;

    Prev_Abs_R = Curr_Abs_R;
    % Prev_Abs_T = Curr_Abs_T;
    Prev_Abs_C = Curr_Abs_C;
end



