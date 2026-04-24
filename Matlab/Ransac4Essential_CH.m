function [finalE, inlierIndx] = Ransac4Essential_CH(PARAMS, matchImg1, matchImg2, K)
    % Initially set maximal number of inliers to 0
    inlierNumMax = 0;

    % Take inverse operation of the intrinsic matrix
    K_inv = inv(K);

    Num_Of_Top_Rank_Ordered_List = ...
        round(PARAMS.TOP_N_RATIO_RANK_ORDERED_LIST * size(matchImg1, 2));
    
    % iterate a fixed number of times for RANSAC
    for i = 1 : PARAMS.RANSAC_ITERATIONS
        gamma1 = zeros(2, 5);
        gamma2 = zeros(2, 5);

        % Select 5 random matches
        idx = randperm(Num_Of_Top_Rank_Ordered_List, 5);
        for j = 1 : 5
            gamma1(1, j) = matchImg1(1, idx(j));
            gamma1(2, j) = matchImg1(2, idx(j));
            gamma2(1, j) = matchImg2(1, idx(j));
            gamma2(2, j) = matchImg2(2, idx(j));
        end

        % Construct the essential matrix E based on the 5 randomly selected points
        E = ComputeEssentialMatrix(gamma1, gamma2, K);

        % Compute coefficients of a line equation
        A = zeros(size(E, 3), length(matchImg1));
        B = zeros(size(E, 3), length(matchImg1));
        C = zeros(size(E, 3), length(matchImg1));
        for j = 1 : size(E, 3)
            calE = K_inv' * E{j} * K_inv;
            A(j, :) = calE(1, :) * matchImg1;
            B(j, :) = calE(2, :) * matchImg1;
            C(j, :) = calE(3, :) * matchImg1;
        end
        
        % Compute the distance from a point to a line for all matches
        dist = zeros(size(E, 3), length(matchImg1));
        denomOfDist = zeros(size(E, 3), length(matchImg1));
        numerOfDist = zeros(size(E, 3), length(matchImg1));
        A_ep = zeros(size(E, 3), length(matchImg1));
        B_it = zeros(size(E, 3), length(matchImg1));
        for k = 1 : length(matchImg1)
            A_ep(:,k) = A(:,k).*matchImg2(1, k);
            B_it(:,k) = B(:,k).*matchImg2(2, k);
        end
        numerOfDist = abs(A_ep + B_it + C);
        denomOfDist = A.^2 + B.^2;
        denomOfDist = sqrt(denomOfDist);
        dist = numerOfDist./denomOfDist;
        
        %> Get maximal inliers among the essential matrix candidates
        for j = 1 : size(E, 3)
            inlierIndx4all = find(dist(j,:) < PARAMS.INLIER_THRESH);
            NumOfInliers = size(inlierIndx4all, 2);
            if (NumOfInliers > inlierNumMax)
                inlierNumMax = NumOfInliers;
                finalE = E{j};
                inlierIndx = inlierIndx4all;
            end
            inlierIndx4all = [];
        end
    end
end