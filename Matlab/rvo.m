clc; clear all; close all;


pe = pyenv;
if pe.Status == "NotLoaded"
    pyenv('Version', 'E:\nvenv\python.exe'); 
end

clear classes;
t_all = tic;
obj = py.importlib.import_module('lg');
py.importlib.reload(obj);

rng(0);


load("MyData\IntrinsicMatrix.mat");


load('MyData/GT_Poses.mat');	%> GT_Poses


mfiledir = fileparts(mfilename('fullpath'));
Image_Sequence = dir(['MyData\fr2_desk\*.png']);
%> Parameters passed to RANSAC
PARAMS.INLIER_THRESH                          = 2;      %> 2 pixels
PARAMS.RANSAC_ITERATIONS                      = 3000;   %> Total number of RANSAC iterations
PARAMS.TOP_N_RATIO_RANK_ORDERED_LIST          = 0.8;    %> Ratio of top rank-ordered list
PARAMS.NUM_OF_FRAMES_FROM_LAST_KF             = 15;
PARAMS.RATIO_OF_COVISIBLE_POINTS_FROM_LAST_KF = 0.55;

%> ==================================================================
%> TODO: Implement a Visual Odometry Pipeline Reducing Motion Drift
%> ==================================================================
%> The world coordinate (origin) is the coordinate of the first frame
Prev_Abs_R = eye(3);
Prev_Abs_T = [0; 0; 0];
Abs_Poses = zeros(3,4,size(Image_Sequence, 1));
Abs_Poses(:,:,1) = [Prev_Abs_R, Prev_Abs_T];
Abs_Cam_Center = zeros(3,size(Image_Sequence, 1));
Abs_Cam_Center(:,1) = -Prev_Abs_R' * Prev_Abs_T;


pz = 0;
tz = 0;
p3z = 0;
yz = 0;

KeyFrame_Indx = [];
Points3D_Cloud = [];
skew_T = @(T)[0, -T(3,1), T(2,1); T(3,1), 0, -T(1,1); -T(2,1), T(1,1), 0];
invK = inv(K);


Previous_Img = imread(fullfile(Image_Sequence(1).folder,Image_Sequence(1).name));
for mi = 2:size(Image_Sequence, 1)
    Current_Img = imread(fullfile(Image_Sequence(mi).folder,Image_Sequence(mi).name));
    

    if mi == 2
        
        t_12 = tic;
        result = py.lg.match_features(fullfile(Image_Sequence(1).folder,Image_Sequence(1).name), fullfile(Image_Sequence(mi).folder,Image_Sequence(mi).name));
        diyier = toc(t_12);
        pz = pz + diyier;


        points0_homo = cell(result{1}.tolist());
        p_h1_1 = double(points0_homo{1});
        p_h1_2 = double(points0_homo{2});
        p_h1_3 = double(points0_homo{3});
        points0_homo = [p_h1_1;p_h1_2;p_h1_3];

        points1_homo = cell(result{2}.tolist());
        p_h2_1 = double(points1_homo{1});
        p_h2_2 = double(points1_homo{2});
        p_h2_3 = double(points1_homo{3});
        points1_homo = [p_h2_1;p_h2_2;p_h2_3];


        

        f1_ranked_12 = points0_homo;
        f2_ranked_12 = points1_homo;
        Prev_f_KF_ranked    = f2_ranked_12;

        
        
        %> Estimate an essential matrix in a RANSAC framework
        tp3 = tic;
        [E, inlierIdx] = Ransac4Essential_CH(PARAMS, f1_ranked_12, f2_ranked_12, K);
        
        %> Recover a veridical relative pose (R,T) from E
        inliers_Img1 = f1_ranked_12(:,inlierIdx);
        inliers_Img2 = f2_ranked_12(:,inlierIdx);
        [Rel_R, Rel_T] = Get_Veridical_RT_from_E(E, inliers_Img1, inliers_Img2, K);
        ttp3 = toc(tp3);
        p3z = p3z + ttp3;
        Prev_f_KF_HavePts3D = inliers_Img2;
        
        %> Triangulation
        Rs = zeros(3,3,1);     Ts = zeros(3,1);
        Rs(:,:,1) = Rel_R;     Ts(:,1) = Rel_T;
        inliers = [inliers_Img1(1:2,:); inliers_Img2(1:2,:)];
        tr = tic;
        Points3D_Cam_Last_KF = Reconstruct_by_LT(Rs, Ts, 2, inliers, K);
        ttr = toc(tr);
        tz =tz+ttr;

        %> Frame 1 is the origin. No need to transform.
        Abs_Poses(:,:,mi) = [Rel_R, Rel_T];
        Abs_Cam_Center(:,mi) = -Rel_R' * Rel_T;
        Abs_R_KF = Rel_R;
        Abs_T_KF = Rel_T;

        Last_Keyframe_Index = 2;
        KeyFrame_Indx = [KeyFrame_Indx, [1,2]];
        Points3D_Cloud = [Points3D_Cloud, Points3D_Cam_Last_KF];

        
    else
        t_e=tic;
        result = py.lg.match_features(fullfile(Image_Sequence(Last_Keyframe_Index).folder,Image_Sequence(Last_Keyframe_Index).name), fullfile(Image_Sequence(mi).folder,Image_Sequence(mi).name));
        meici = toc(t_e);
        pz = pz +meici;
        disp(['Each match: ', num2str(meici), ' Sec']);

        mi

        points0_homo = cell(result{1}.tolist());
        p_h1_1 = double(points0_homo{1});
        p_h1_2 = double(points0_homo{2});
        p_h1_3 = double(points0_homo{3});
        points0_homo = [p_h1_1;p_h1_2;p_h1_3];

        points1_homo = cell(result{2}.tolist());
        p_h2_1 = double(points1_homo{1});
        p_h2_2 = double(points1_homo{2});
        p_h2_3 = double(points1_homo{3});
        points1_homo = [p_h2_1;p_h2_2;p_h2_3];
         
        f_KF_ranked           = points0_homo;
        f_CF_ranked           =points1_homo;

        %> Find covisible feature index
        [~, ~, CovIndx_KF_ranked] = intersect(Prev_f_KF_ranked', f_KF_ranked', 'rows');
        f_KF_ranked_ = f_KF_ranked(:,CovIndx_KF_ranked);
        f_CF_ranked_ = f_CF_ranked(:,CovIndx_KF_ranked);
        [~, Prev_KF_Indx_HavePts3D, f_CF_Indx_HavePts3D] = intersect(Prev_f_KF_HavePts3D', f_KF_ranked_', 'rows');
        f_KF_HavePts3D = f_KF_ranked_(:,f_CF_Indx_HavePts3D);
        f_CF_HavePts3D = f_CF_ranked_(:,f_CF_Indx_HavePts3D);


        %> Estimate absolute camera pose. This pose is under the first camera coordinate.
        Points3D_Last_KF_Cam1 = Points3D_Cam_Last_KF(:,Prev_KF_Indx_HavePts3D);
        tp3 = tic;
        [Abs_R_CF, Abs_T_CF, inlierRansacIndx, you] = Msac4AbsolutePose_CHM(PARAMS, Points3D_Last_KF_Cam1, f_CF_HavePts3D, K);
        ttp3 = toc(tp3);
        yz = yz +you;
        p3z = p3z+ttp3;
        Abs_Poses(:,:,mi) = [Abs_R_CF, Abs_T_CF];
        Abs_Cam_Center(:,mi) = -Abs_R_CF' * Abs_T_CF;
        f_CF_HavePts3D = f_CF_HavePts3D(:,inlierRansacIndx);
        Points3D_Last_KF_Cam1 = Points3D_Last_KF_Cam1(:,inlierRansacIndx);

        %> ================================================================
        %> Keyframe Selection and Triangulating New Observations if Needed
        %> ================================================================
        Num_Of_Frames_From_Last_Keyframe = mi - Last_Keyframe_Index;
        Ratio_Of_Covisible_Points = length(Prev_KF_Indx_HavePts3D) / size(Points3D_Cam_Last_KF,2);

        %> Criterion for deciding the current frame as a keyframe
        if Num_Of_Frames_From_Last_Keyframe >= PARAMS.NUM_OF_FRAMES_FROM_LAST_KF || ...
           Ratio_Of_Covisible_Points < PARAMS.RATIO_OF_COVISIBLE_POINTS_FROM_LAST_KF

            %> For a KF, triangulate newly observed feature matches
            %> (i) Find the newly observed feature correspondences
            [~, f_CF_Indx_HavePts3D, ~] = intersect(f_CF_ranked', f_CF_HavePts3D', 'rows');
            f_CF_ranked_transpose = f_CF_ranked';
            f_KF_ranked_transpose = f_KF_ranked';
            f_CF_ranked_transpose(f_CF_Indx_HavePts3D,:) = [];
            f_KF_ranked_transpose(f_CF_Indx_HavePts3D,:) = [];
            f_CF_ranked_HaveNoPts3D = f_CF_ranked_transpose';
            f_KF_ranked_HaveNoPts3D = f_KF_ranked_transpose';

            %> (ii) Calculate the relative pose of the current frame w.r.t.
            %      the previous keyframe
            Rel_R_wrt_Last_KF = Abs_R_CF * Abs_R_KF';
            Rel_T_wrt_Last_KF = Abs_T_CF - Abs_R_CF * Abs_R_KF' * Abs_T_KF;

            %> (iii) Find inliers from the newly observed feature correspondences
            E = skew_T(Rel_T_wrt_Last_KF) * Rel_R_wrt_Last_KF;
            F = invK' * E * invK;
            Apixel = F(1,:) * f_KF_ranked_HaveNoPts3D;
            Bpixel = F(2,:) * f_KF_ranked_HaveNoPts3D;
            Cpixel = F(3,:) * f_KF_ranked_HaveNoPts3D;
            A_xi  = Apixel.*f_CF_ranked_HaveNoPts3D(1,:);
            B_eta = Bpixel.*f_CF_ranked_HaveNoPts3D(2,:);
            numerOfDist = abs(A_xi + B_eta + Cpixel);
            denomOfDist = Apixel.^2 + Bpixel.^2;
            dist2EL = numerOfDist./sqrt(denomOfDist);
            inlier_Indx_KC = find(dist2EL <= PARAMS.INLIER_THRESH);
            
            f_CF_Inliers_HaveNoPts3D = f_CF_ranked_HaveNoPts3D(:,inlier_Indx_KC);
            f_KF_Inliers_HaveNoPts3D = f_KF_ranked_HaveNoPts3D(:,inlier_Indx_KC);

            %> (iv) Triangulate. 3D points are under the coordinate of last keyframe
            Rs = zeros(3,3,1);             Ts = zeros(3,1);
            Rs(:,:,1) = Rel_R_wrt_Last_KF; Ts(:,1) = Rel_T_wrt_Last_KF;
            KF_CF_Inliers = [f_KF_Inliers_HaveNoPts3D(1:2,:); f_CF_Inliers_HaveNoPts3D(1:2,:)];
            tr = tic;
            Points3D_Cam_Last_KF = Reconstruct_by_LT(Rs, Ts, 2, KF_CF_Inliers, K);
            ttr = toc(tr);
            tz = tz + ttr;

            %> (v) Transform 3D points from last keyframe coordinate to the first camera coordinate
            Points3D_Cam1_New = Abs_R_KF' * (Points3D_Cam_Last_KF - Abs_T_KF);
            Points3D_Cam_Last_KF = [Points3D_Cam1_New, Points3D_Last_KF_Cam1];
            Points2D_CF          = [f_CF_Inliers_HaveNoPts3D, f_CF_HavePts3D];
            Points3D_Cloud       = [Points3D_Cloud, Points3D_Cam_Last_KF];

            %> (vi) Mark the current frame as a keyframe
            KeyFrame_Indx = [KeyFrame_Indx, mi];
            Last_Keyframe_Index = mi;
            % f_KF = f_CF;
            % d_KF = d_CF;
            Abs_R_KF = Abs_R_CF;
            Abs_T_KF = Abs_T_CF;
            Prev_f_KF_ranked = f_CF_ranked;
            Prev_f_KF_HavePts3D = Points2D_CF;
        end
    end

    %> Monitor the progress
    if mod(mi, 5) == 0, fprintf(". "); end
    if mod(mi, 50) == 0, fprintf("\n"); end
end
fprintf("\n");

%> Scale up/down according to the ground truth pose
GT_c1 = -GT_Poses(:,1:3,1)' * GT_Poses(:,4,1);
GT_c2 = -GT_Poses(:,1:3,2)' * GT_Poses(:,4,2);
scale = norm(GT_c1 - GT_c2);
Estimated_Poses = zeros(size(Abs_Poses));
for ci = 1:size(Abs_Poses, 3)
    Estimated_Poses(:,:,ci) = [Abs_Poses(:,1:3,ci), Abs_Poses(:,4,ci).*scale];
end

%> Finally, visualize the trajectory
Visualize_Trajectory(GT_Poses, Estimated_Poses, KeyFrame_Indx);
zong = toc(t_all);
disp(['1-2 frames: ', num2str(diyier), ' sec']);
disp(['Total duration: ', num2str(zong), ' sec']);

% % disp('lg:')
% % mean(IRL)
s = size(GT_Poses, 3);

RMSER = 0;

for i = 1:s
    r = Estimated_Poses(:,1:3,i);
    rg = GT_Poses(:, 1:3, i);
    re = (acos(0.5 * (trace(rg' * r) - 1)))^2;

    RMSER = RMSER + re;

end

RMSER = sqrt(RMSER/s);

RMSET_standard = 0;
for i = 1:s % Standard APE for translation is typically calculated from the first frame
    t_est = Estimated_Poses(:,4,i);
    t_gt = GT_Poses(:, 4, i);

    te_squared = sum((t_gt - t_est).^2); % This is norm(t_gt - t_est)^2

    RMSET_standard = RMSET_standard + te_squared;
end

RMSET_standard = sqrt(RMSET_standard/s);

disp("Standard RMSE for translations (Euclidean distance) (without KA):");
disp(RMSET_standard);



disp("RMSE for rotations (without KA):");
disp(RMSER);





