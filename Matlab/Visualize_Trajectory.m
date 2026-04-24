function Visualize_Trajectory(GT_Poses, Estimated_Poses, KeyFrame_Indx)

%> Code Description: 
%     Given a series of estimated and ground truth poses of the same scale, 
%     plot the camera trajectory.
%
%> Inputs: 
%     GT_Poses:         A 3x4xN matrix of N ground truth poses. For image i,
%                       GT_Poses(:,1:3,i) is the rotation matrix while 
%                       GT_Poses(:,4,i) is the translation vector.
%     Estimated_Poses:  A matrix of the same size and structure as
%                       GT_Poses. Note that the estimated poses are already 
%                       scaled up/down according to the groud truth poses.
%     KeyFrame_Indx:    A list of keyframe indices. If there are no keyframe
%                       indices, set every 10 frames as your keyframes.
%
%> Outputs:
%     None
%
%> (c) LEMS, Brown University
%> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
%> Feb. 15th, 2024

    %> ================================================================
    %> Plot GT Trajectory
    %> ================================================================
    Abs_Cam_Center_GT = zeros(3,size(GT_Poses, 3));
    for ci = 1:size(GT_Poses, 3)
        Abs_Cam_Center_GT(:,ci) = -GT_Poses(1:3,1:3,ci)'*GT_Poses(1:3,4,ci);
    end

    figure;
    plot3(Abs_Cam_Center_GT(1,:), Abs_Cam_Center_GT(2,:), Abs_Cam_Center_GT(3,:), 'bo-'); hold on;
    plotCamera('Location', [0;0;0], 'Orientation', eye(3), 'Opacity', 0.3, 'Size', 0.02, 'Color', [0,0,1]); hold on;
    for ci = 1:size(GT_Poses, 3)
        R = GT_Poses(:,1:3,ci);
        C = Abs_Cam_Center_GT(:,ci);
        if ~isempty(find(KeyFrame_Indx == ci))
            plotCamera('Location', C, 'Orientation', R, 'Opacity', 0.3, 'Size', 0.02, 'Color', [0,0,1]);
        end
        hold on;
    end

    %> ================================================================
    %> Plot Estimated Trajectory
    %> ================================================================
    Abs_Cam_Center = zeros(3,size(Estimated_Poses, 3));
    for ci = 1:size(Estimated_Poses, 3)
        Abs_Cam_Center(:,ci) = -Estimated_Poses(1:3,1:3,ci)'*Estimated_Poses(1:3,4,ci);
    end
    
    plot3(Abs_Cam_Center(1,:), Abs_Cam_Center(2,:), Abs_Cam_Center(3,:), 'go-'); hold on;
    plotCamera('Location', [0;0;0], 'Orientation', eye(3), 'Opacity', 0.3, 'Size', 0.02, 'Color', [0,1,0]); hold on;
    for ci = 1:size(Estimated_Poses, 3)
        R = Estimated_Poses(:,1:3,ci);
        C = Abs_Cam_Center(:,ci);
        if ~isempty(find(KeyFrame_Indx == ci))
            plotCamera('Location', C, 'Orientation', R, 'Opacity', 0.3, 'Size', 0.02, 'Color', [0, 1, 0]);
        end
        hold on;
    end

    xlabel('x');
    ylabel('y');
    zlabel('z');
    axis equal;
    legend({"Ground Truth Trajectory", "Estimated Trajectory without keyponts adjustment"});
    grid on;
    set(gcf,'color','w');
    set(gca,'FontSize',15);

end
