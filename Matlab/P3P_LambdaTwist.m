function [Rs, Ts] = P3P_LambdaTwist(Points2D, Points3D)

%> Code Description: 
%     Given 3 pairs of 3D-2D correspondences, return all valid absolute
%     poses using lambda twist P3P solver.
%
%> Inputs: 
%     Points2D:  A 3x3 matrix where each column, Points2D(:,i) is a 2D
%                feature location in meters.
%     Points3D:  A 3x3 matrix where each column, Points3D(:,i) is a 3D
%                location of point i corresponding to the 2D feature 
%                Points2D(:,i). 
%
%> Outputs:
%     Rs:        N valid rotation matrices structured by Rs(3,3,N).
%     Ts:        N valid translation vectors structed by Ts(3,N).
%
%> Reference:
%   M. Persson and K. Nordberg, "Lambda Twist: An Accurate Fast Robust
%   Perspective Three Point (P3P) Solver," ECCV2018.
%   https://github.com/midjji/pnp/blob/master/lambdatwist/lambdatwist.p3p.h
%
%> (c) NEC Corporation and Brown University
%> Gaku Nakano and Chiang-Heng Chien

    %> Normalize the length of ys
    y1 = Points2D(:,1) / norm(Points2D(:,1));
    y2 = Points2D(:,2) / norm(Points2D(:,2));
    y3 = Points2D(:,3) / norm(Points2D(:,3));

    b12 = -2*y1'*y2;
    b13 = -2*y1'*y3;
    b23 = -2*y2'*y3;
    
    x1 = Points3D(:,1);
    x2 = Points3D(:,2);
    x3 = Points3D(:,3);
    
    d12 = x1 - x2;
    d13 = x1 - x3;
    d23 = x2 - x3;
    d12xd13 = cross(d12, d13);
    
    a12 = norm(d12)^2;
    a13 = norm(d13)^2;
    a23 = norm(d23)^2;

    % a*g^3 + b*g^2 + c*g + d = 0
    c31  = -0.5*b13;
    c23  = -0.5*b23;
    c12  = -0.5*b12;
    blob = c12*c23*c31 - 1;

    s31_squared = 1 - c31*c31;
    s23_squared = 1 - c23*c23;
    s12_squared = 1 - c12*c12;

    p3 = (a13*(a23*s31_squared - a13*s23_squared));
    p2 = 2.0*blob*a23*a13 + a13*(2.0*a12 + a13)*s23_squared + a23*(a23 -     a12)*s31_squared;
    p1 = a23*(a13 - a23)*s12_squared - a12*a12*s23_squared - 2.0*a12*(blob*a23 +    a13*s23_squared);
    p0 = a12*(a12*s23_squared - a23*s12_squared);    

    % p3 is essentially det(D2) so it's definitely > 0; otherwise, it's degenracy
    if abs(p3)>=abs(p0)
        p3 = 1.0/p3;
        p2 = p3*p2;
        p1 = p3*p1;
        p0 = p3*p0;

        % get sharpest real root of above...
        g = cubick(p2,p1,p0);
    else

        % lower numerical performance
        g = 1 / cubick(p1/p0,p2/p0,p3/p0);
    end

    %> Swap D1,D2 and the coeffs!
    % oki, Ds are:
    % D1=M12*XtX(2,2) - M23*XtX(1,1);
    % D2=M23*XtX(3,3) - M13*XtX(2,2);
    % [    a23 - a23*g,                 (a23*b12)/2,              -(a23*b13*g)/2]
    % [    (a23*b12)/2,           a23 - a12 + a13*g, (a13*b23*g)/2 - (a12*b23)/2]
    % [ -(a23*b13*g)/2, (a13*b23*g)/2 - (a12*b23)/2,         g*(a13 - a23) - a12]
    A00 = a23*(1.0 - g);
    A01 = (a23*b12)*0.5;
    A02 = (a23*b13*g)*(-0.5);
    A11 = a23 - a12 + a13*g;
    A12 = b23*(a13*g - a12)*0.5;
    A22 = g*(a13 - a23) - a12;
    A = [A00, A01, A02
         A01, A11, A12
         A02, A12, A22];
    
    % get sorted eigenvalues and eigenvectors given that one should be zero...
    [V, L] = eigwithknown0(A);
    v = sqrt(max(0, -L(2)/L(1)));
    
    valid = 0;
    Ls    = zeros(3,4);
    
    % use the t = Vl with t2,st2,t3 and solve for t3 in t2
	% +v
	s = v;
	w2 = 1/( s*V(1,2) - V(1,1));
	w0 = (V(2,1) - s*V(2,2))*w2;
	w1 = (V(3,1) - s*V(3,2))*w2;

	a = 1 / ((a13 - a12)*w1*w1 - a12*b13*w1 - a12);
	b = ( a13*b12*w1 - a12*b13*w0 - 2*w0*w1*(a12 - a13) )*a;
	c = ( (a13 - a12)*w0*w0 + a13*b12*w0 + a13 )*a;

	if( b*b - 4*c >= 0 )
        [tau1, tau2] = root2real(b, c);
		if tau1>0
			tau = tau1;
			d = a23/(tau*(b23 + tau) + 1);
			l2 = sqrt(d);
			l3 = tau*l2;
			l1 = w0*l2 + w1*l3;
			if(l1 >= 0)
				valid = valid + 1;
                Ls(:,valid) = [l1; l2; l3];
			end
		end
		if tau2>0
			tau = tau2;
			d = a23/(tau*(b23 + tau) + 1);
			l2 = sqrt(d);
			l3 = tau*l2;
			l1 = w0*l2 +w1*l3;
			if(l1 >= 0)
				valid = valid + 1;
                Ls(:,valid) = [l1; l2; l3];
			end
		end
	end
    

    %+v
	s = -v;
	w2 = 1/( s*V(1,2) - V(1,1));
	w0 = (V(2,1) - s*V(2,2))*w2;
	w1 = (V(3,1) - s*V(3,2))*w2;

	a = 1 / ((a13 - a12)*w1*w1 - a12*b13*w1 - a12);
	b = ( a13*b12*w1 - a12*b13*w0 - 2*w0*w1*(a12 - a13) )*a;
	c = ( (a13 - a12)*w0*w0 + a13*b12*w0 + a13 )*a;

    
    if( b*b - 4*c >= 0)
        [tau1, tau2] = root2real(b, c);
        if tau1>0
            tau = tau1;
            d = a23/(tau*(b23 + tau) + 1);
            l2 = sqrt(d);
            l3 = tau*l2;
            l1 = w0*l2 +w1*l3;
            if(l1>= 0)
				valid = valid + 1;
                Ls(:,valid) = [l1; l2; l3];
            end
        end
        if tau2>0
            tau = tau2;
            d = a23/(tau*(b23 + tau) + 1);
            l2 = sqrt(d);
            l3 = tau*l2;
            l1 = w0*l2 +w1*l3;
            if(l1 >= 0)
				valid = valid + 1;
                Ls(:,valid) = [l1; l2; l3];
            end
        end
    end

    for i = 1:valid
        refinement_iterations = 5;
        Ls(:,i) = gauss_newton_refineL(Ls(:,i),a12,a13,a23,b12,b13,b23,refinement_iterations);
    end

    R = zeros(3,3,valid);
    t = zeros(3,valid);
    X = [d12, d13, d12xd13];
    X = inv(X);
    
    for i = 1:valid
        % compute the rotation:
        ry1 = y1*Ls(1,i);
        ry2 = y2*Ls(2,i);
        ry3 = y3*Ls(3,i);
        
        yd1 = ry1 - ry2;
        yd2 = ry1 - ry3;
        yd1xd2 = cross(yd1, yd2);
        
        Y = [yd1, yd2, yd1xd2];
        
        R(:,:,i) = Y*X;
        t(:,i)   = ry1 - R(:,:,i)*x1;
        
    end

    %> Return as outputs
    Rs = R;
    Ts = t;
end


function [r1, r2] = root2real(b, c)
    v = b*b - 4.0*c;
    if v < 0
        r1 = 0.5*b;
        r2 = r1;
    else
        y = sqrt(v);
        if(b<0)
            r1 = 0.5*(-b+y);
            r2 = 0.5*(-b-y);
        else
            r1 = 2.0*c/(-b+y);
            r2 = 2.0*c/(-b-y);
        end
    end
end

function r0 = cubick(b, c, d)

    % Choose initial solution
    % not monotonic
    if b*b >= 3*c
        % h has two stationary points, compute them
        %t1 = t - sqrt(diff);
        v=sqrt(b*b - 3*c);
        t1 = (-b - v)/3;

        % Check if h(t1) > 0, in this case make a 2-order approx of h around t1
        k = ( (t1+b)*t1+c )*t1 + d;

        if k > 0
            %Find leftmost root of 0.5*(r0 -t1)^2*(6*t1+2*b) +  k = 0
            r0 = t1 - sqrt(-k/(3*t1 + b));
            % or use the linear comp too
            %r0=t1 -
        else
            t2 = (-b + v)/3;
            k = ((t2+b)*t2+c)*t2+d;
            %Find rightmost root of 0.5*(r0 -t2)^2*(6*t2+2*b) +  k1 = 0
            r0 = t2 + sqrt(-k/(3.0*t2 + b));
        end

    else
        r0 = -b/3;
        if abs( (3*r0+2*b)*r0+c ) < 1e-4
            r0 = r0 + 1;
        end
    end



    % Do ITER Newton-Raphson iterations
    % Break if position of root changes less than 1e-16
    KLAS_P3P_CUBIC_SOLVER_ITER = 50;
    for cnt = 1:KLAS_P3P_CUBIC_SOLVER_ITER

        %(+ (* r0 (+  c (* (+ r0 b) r0) )) d )
        fx=(((r0+b)*r0+c)*r0+d);


        if cnt<7 || abs(fx)>eps
            fpx = (3*r0+2*b)*r0 + c;
            r0  = r0 - fx/fpx;
        else
            break
        end
    end

    
end


% eigen decomp of a matrix which has a 0 eigen value
% x 3x3 matrix
% E eigenvectors
% L eigenvalues
function [E, L] = eigwithknown0(x)

    L = zeros(3,1);

    % one eigenvalue is known to be 0.
    %the known one...
    L(3) = 0;
    v3 = [x(4)*x(8)- x(7)*x(5)
          x(7)*x(2)- x(8)*x(1)
          x(5)*x(1)- x(4)*x(2)];


    v3 = v3 / norm(v3);


    x01_squared = x(1,2)^2;
    % get the two other...
    b = - x(1,1) - x(2,2) - x(3,3);
    c = - x01_squared - x(1,3)^2 - x(2,3)^2 + x(1,1)*(x(2,2) + x(3,3)) + x(2,2)*x(3,3);

    % roots(poly(x))
    [e1, e2] = root2real(b,c);

    if abs(e1) < abs(e2)
        [e2, e1] = deal(e1,e2);
    end
    L(1) = e1;
    L(2) = e2;



    mx0011 = -x(1,1)*x(2,2);
    prec_0 =  x(1,2)*x(2,3) - x(1,3)*x(2,2);
    prec_1 =  x(1,2)*x(1,3) - x(1,1)*x(2,3);


    e=e1;
    tmp=1.0/(e*(x(1,1) + x(2,2)) + mx0011 - e*e + x01_squared);
    a1= -(e*x(1,3) + prec_0)*tmp;
    a2= -(e*x(2,3) + prec_1)*tmp;
    rnorm=1.0/sqrt(a1*a1 +a2*a2 + 1.0);
    a1 = a1*rnorm;
    a2 = a2*rnorm;
    v1 = [a1; a2; rnorm];

    %e=e2;
    tmp2=1.0/(e2*(x(1,1) + x(2,2)) + mx0011 - e2*e2 + x01_squared);
    a21= -(e2*x(1,3) + prec_0)*tmp2;
    a22= -(e2*x(2,3) + prec_1)*tmp2;
    rnorm2=1.0/sqrt(a21*a21 +a22*a22 +1.0);
    a21 = a21*rnorm2;
    a22 = a22*rnorm2;
    v2 = [a21; a22; rnorm2];


    % optionally remove axb from v1,v2
    % costly and makes a very small difference!
    % v1=(v1-v1.dot(v3)*v3);v1.normalize();
    % v2=(v2-v2.dot(v3)*v3);v2.normalize();
    % v2=(v2-v1.dot(v2)*v2);v2.normalize();
    E = [v1, v2, v3];

end


function L = gauss_newton_refineL(L, a12, a13, a23, b12, b13, b23, iterations)

    % consexpr makes ieasier for the compiler to unroll
    for i = 1:iterations
        l1 = L(1);
        l2 = L(2);
        l3 = L(3);
        r1 = l1*l1 + l2*l2 + b12*l1*l2 - a12;
        r2 = l1*l1 + l3*l3 + b13*l1*l3 - a13;
        r3 = l2*l2 + l3*l3 + b23*l2*l3 - a23;

        if abs(r1) + abs(r2) + abs(r3) < 1e-10
            break
        end



        dr1dl1 = 2*l1 +b12*l2;
        dr1dl2 = 2*l2 +b12*l1;

        dr2dl1 = 2*l1 +b13*l3;
        dr2dl3 = 2*l3 +b13*l1;


        dr3dl2 = 2*l2 + b23*l3;
        dr3dl3 = 2*l3 + b23*l2;


        r = [r1; r2; r3];

        % or skip the inverse and make iexplicit...
        
        v0  = dr1dl1;
        v1  = dr1dl2;
        v3  = dr2dl1;
        v5  = dr2dl3;
        v7  = dr3dl2;
        v8  = dr3dl3;
        det = 1/(- v0*v5*v7 - v1*v3*v8);

        Ji = [-v5*v7, -v1*v8,  v1*v5
              -v3*v8,  v0*v8, -v0*v5
               v3*v7, -v0*v7, -v1*v3];
        L1 = L - det*(Ji*r);

        %%l=l - g*H\G;%inv(H)*G
        %L=L - g*J\r; %% works because the size is ok!

        l1 = L(1);
        l2 = L(2);
        l3 = L(3);
        r11 = l1*l1 + l2*l2 +b12*l1*l2 -a12;
        r12 = l1*l1 + l3*l3 +b13*l1*l3 -a13;
        r13 = l2*l2 + l3*l3 +b23*l2*l3 -a23;
        if abs(r11) +abs(r12) + abs(r13) > abs(r1) +abs(r2) +abs(r3)
            break
        else
            L = L1;
        end

    end


end