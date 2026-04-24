function [Rel_R, Rel_T] = Get_Veridical_RT_from_E(E, inliers_Img1, inliers_Img2, K)
    
    [U, ~, V] = svd(E);
    W = [0,-1,0;1,0,0;0,0,1];
    R1 = U*W*V';
    R2 = U*W'*V';
    T1 = U(:,3);
    T2 = -U(:,3);
    
    if det(R1) < 0 || det(R2) < 0
        E = E*(-1);
        [U, ~, V] = svd(E);
        R1 = U*W*V';
        R2 = U*W'*V';
        T1 = U(:,3);
        T2 = -U(:,3);
    end
    
    I = eye(3);
    
    Rall{1} = R1;
    Rall{2} = R1;
    Rall{3} = R2;
    Rall{4} = R2;
    Tall{1} = T1;
    Tall{2} = T2;
    Tall{3} = T1;
    Tall{4} = T2;
    e1 = [1;0;0];
    e3 = [0;0;1];
    
    invK               = inv(K);
    Gamma_1            = invK*inliers_Img1;
    Gamma_2            = invK*inliers_Img2;
    Num_Positive_Depths   = [];
    
    for(idx = 1:4)
        R12  = Rall{idx};
        T12  = Tall{idx};
        rho1 = (e1'*T12 - (e3'*T12)*(e1'*Gamma_2)) ./ ...
               ((e3'*R12*Gamma_1).*(e1'*Gamma_2) - (e1'*R12*Gamma_1));
        rho2   = ((e1'*T12)*(e3'*R12*Gamma_1) - (e3'*T12)*(e1'*R12*Gamma_1)) ./ ...
               ((e3'*R12*Gamma_1) .* (e1'*Gamma_2) - (e1'*R12*Gamma_1));
        
        Num_Positive_Depths = [Num_Positive_Depths; sum(rho1>0) + sum(rho2>0)];
    end
    
    [~, maxidx] = max(Num_Positive_Depths);
    Rel_R = Rall{maxidx};
    Rel_T = Tall{maxidx};
end