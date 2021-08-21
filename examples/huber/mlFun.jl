#====================================================================
 Different Functions About Parameter Log-Likelihood Objective
 funcV: function value for the total objective
 gradH: gradient and hessian of huber objective w.r.t. κ
 ntSolver: newton solver for huber objective
====================================================================#
#--------------------------------------------------------------------
# Function Value
#--------------------------------------------------------------------
function funcV(r::Array{Float64,1},κ::Float64)
	m = length(r);
	# function value of huber
	val = 0.0;
	for i = 1:m
		if abs(r[i]) <= κ
			val += 0.5*r[i]^2;
		else
			val += κ*abs(r[i]) - 0.5*κ^2;
		end
	end
	val /= m;
	# log of constant function w.r.t. κ
	val += log(2.0/κ*exp(-0.5*κ^2)+sqrt(2.0*pi)*erf(κ/sqrt(2.0)));
	return val
end
#--------------------------------------------------------------------
# Gradient and Hessian Function
#--------------------------------------------------------------------
function gradH(r::Array{Float64,1},κ::Float64)
	m = length(r);
	# gradient and hessian for huber penalty w.r.t. κ
	grad1 = 0.0;
	H1    = 0.0;
	for i = 1:m
		if abs(r[i]) > κ
			grad1 += abs(r[i]) - κ;
			H1    -= 1.0;
		end
	end
	grad1 /= m;
	H1    /= m;
	# gradient and hessian for normalization function
	expκ  = exp(-0.5*κ^2);
	c     = 2.0*expκ/κ + sqrt(2.0*pi)*erf(κ/sqrt(2.0));
	cp    = -2.0*expκ/κ^2;			# c'(κ)
	cpp   = (4.0/κ^3+2.0/κ)*expκ;	# c''(κ)
	grad2 = cp/c;					# [log(c(κ))]'
	H2    = (cpp*c-cp^2)/c^2;		# [log(c(κ))]''
	# return the gradient and hessian
	grad  = grad1+ grad2;
	H     = H1 + H2;
	# return grad-1.0/κ^2, H+2.0/κ^3
	return grad, H
end
#--------------------------------------------------------------------
# Newton Method Solver
#--------------------------------------------------------------------
function ntSolver(r::Array{Float64,1},κ::Float64,
					  tol::Float64,itermax::Int64)
	iter = 0;
	η    = 0.01;
	g,H  = gradH(r,κ);
	err  = norm(g);
	# print(g,"\n");
		while err >= tol
			κ    -= g/H;
			#κ    -= η*g;
			iter += 1;
			if iter % 100 == 0
				@printf("κiter %d, err %1.5e\n",iter,err);
			end
			if iter > itermax
				print("Warning Reach Maximum Iter Number!\n");
				break;
			end
			g,H = gradH(r,κ);
			err = norm(g);
		end
		# @printf("κiter %d, err %1.5e\n",iter,err);
	return κ
end