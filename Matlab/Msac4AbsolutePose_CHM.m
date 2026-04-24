function [Abs_R_y, Abs_T_y, inlierIndx, t] = Ransac4AbsolutePose_CHM(PARAMS, Points3D, Points2D, K)
    
    minTotalCost = Inf;
    R_max_support = eye(3);
    T_max_support = zeros(1,3);
    inlier_index_max_support = []; 
    
    %> Convert 2D points from pixels to meters
    Points2D_metric = inv(K) * Points2D;

    T_thresh = PARAMS.INLIER_THRESH; 

    %> Iterate over a fixed number of RANSAC iterations
    for i = 1 : PARAMS.RANSAC_ITERATIONS
        gamma = zeros(3, 3);
        Gamma = zeros(3, 3);
        %> Randomly select 3 random 3D-2D matches
        idx = randperm(size(Points2D, 2), 3);
        for j = 1 : 3
            gamma(:,j) = Points2D_metric(:,idx(j));
            Gamma(:,j) = Points3D(:,idx(j));
        end
        %> Get absolute pose from P3P lambda twist algorithm
        [Rs, Ts] = P3P_LambdaTwist(gamma, Gamma);
        %> Find number of inliers from reprojection errors
        for ci = 1:size(Rs, 3)
            R_ = Rs(:,:,ci);
            T_ = Ts(:,ci);
            Reproj_Points2D = K*(R_*Points3D + T_);
            Reproj_Points2D = Reproj_Points2D ./ Reproj_Points2D(3,:);
            %> Calculate the reprojection errors
            Reproj_Error = vecnorm(Reproj_Points2D - Points2D, 2, 1);
            
            
            Costs = Reproj_Error;
            Costs(Costs > T_thresh) = T_thresh;
            currentTotalCost = sum(Costs);
            

            
            if currentTotalCost < minTotalCost
                minTotalCost = currentTotalCost;
                R_max_support = R_;
                T_max_support = T_;
                inlier_index_max_support = find(Reproj_Error < T_thresh);
            end
        end
    end
    
    %> Return as outputs
    Abs_R = R_max_support;
    Abs_T = T_max_support;
    inlierIndx = inlier_index_max_support;

    % 检查 RANSAC 是否找到了任何内点 (保留原始框架)
    if isempty(inlierIndx)
        disp('RANSAC 未找到足够的内点，优化跳过。');
        Abs_R_y = Abs_R;
        Abs_T_y = Abs_T;
        t = 0;
        return;
    end
    
    p33 = Points3D(:,inlierIndx);
    p22 = Points2D(:,inlierIndx);
    x0 = zeros(1,6);
    x0(1:3) = rotm2eul(Abs_R);
    x0(4:6) = Abs_T';
    tic;
    ff = @(x)(error(x, p33,p22, K)); % error 函数现在在下面被修改了
    options = optimoptions('lsqnonlin');
    options.Algorithm = 'levenberg-marquardt';
    options.Display = 'iter'; % 关闭显示
    
    x00 = lsqnonlin(ff, x0, [], [], options);
    youhua = toc;
    disp(['每次优化: ', num2str(youhua), ' 秒']);
    t = youhua;
    Abs_R_y = eul2rotm(x00(1:3));
    Abs_T_y = x00(4:6)';
end

% ---
% [关键修复] 修正 error 函数，使其返回 2N x 1 的残差向量
% 这样 lsqnonlin 就能正确最小化误差的平方和
% ---
function E = error(x, p3, p2, k)
    r1 = x(1:3);
    R1 = eul2rotm(r1);
    T1 = x(4:6)';
    
    s = size(p3,2);
    E = zeros(2 * s, 1); % 预分配 2*N x 1 的残差向量

    for i = 1:s
        pp3 = p3(:,i);
        ga = R1*pp3 +T1;
        c = k*ga;
        c = c./c(end);
        pp2 = p2(:,i);
        
        % [修复] 填充残差向量，而不是求和
        % ee = sqrt((pp2(1,1) - c(1,1))^2 + (pp2(2,1) - c(2,1))^2);
        % EE = EE + ee;
        
        E( (i-1)*2 + 1 ) = pp2(1,1) - c(1,1); % x 轴残差
        E( (i-1)*2 + 2 ) = pp2(2,1) - c(2,1); % y 轴残差
    end
    % [修复] 移除 E = EE; lsqnonlin 将自动最小化 sum(E.^2)
end