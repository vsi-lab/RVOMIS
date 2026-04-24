function Gamma_w = Reconstruct_by_LT(Rs, Ts, N, inliers, K)
    
    K_inv = inv(K);
    I = eye(3);
    skew_T = @(T)[0, -T(3,1), T(2,1); T(3,1), 0, -T(1,1); -T(2,1), T(1,1), 0];
    
    Rel_Rs = cell(1,N);
    Rel_Ts = cell(1,N);
    Rel_Cs = cell(1,N);
    Rel_Rs{1} = eye(3);
    Rel_Ts{1} = [0; 0; 0];
    for i = 2:N
        Rel_Rs{i} = Rs(:,:,i-1);
        Rel_Ts{i} = Ts(:,i-1);
    end

    %> Loop over all points
    Gamma_w = zeros(3,size(inliers, 2));
    for i = 1:size(inliers, 2)
        A = [];
        offset = 1;
        for v = 1:N
            gamma_i = K_inv * [inliers(offset:offset+1,i); 1];
            skew_gamma = skew_T(gamma_i);
            Ri = Rel_Rs{v};
            Ti = Rel_Ts{v};
            A = [A; skew_gamma*[Ri, Ti]];
            offset = offset + 2;
        end
        [~, ~, V] = svd(A);
        p3d = V(:,end);
        p3d = p3d ./ p3d(end);
        Gamma_w(:,i) = p3d(1:3,1);
    end
    
end