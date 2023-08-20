using CSV,DataFrames,StatsBase,Statistics,LinearAlgebra,NLopt
using Printf,Plots,Dates

df = DataFrame(CSV.File("D:\\Julia\\data.csv";select=["000852.SH","H11008.CSI","AU9999.SGE","CBA00301.CS"]))
dropmissing(df)
name = names(df)
X = Matrix(df)
dfcov = cov(X)

function risk_budget_objective(weights::Vector,grad::Vector)
    cov = dfcov
    sigma = sqrt.(weights'*(cov'*weights))
    MRC = (cov'*weights)./sigma
    TRC = weights .* MRC
    grad = zeros(4)
    if length(grad) > 0
        grad[1] = 2.0 * (sum(TRC)- TRC[1])
        grad[2] = 2.0 * (sum(TRC)- TRC[2])
        grad[3] = 2.0 * (sum(TRC)- TRC[3])
        grad[4] = 2.0 * (sum(TRC)- TRC[4])
    end
    delta_TRC = [sum((i .- TRC).^2) for i in TRC]
    return sum(delta_TRC)
end

function total_weight_constraint(x::Vector, grad::Vector)
    grad = [-1.0,-1.0,-1.0,-1.0]
    return sum(x) - 1.0
end

opt = Opt(:LN_COBYLA,4)
opt.lower_bounds = [0.0,0.0,0.0,0.0]
opt.ftol_abs = 1e-20
opt.min_objective = risk_budget_objective
equality_constraint!(opt,total_weight_constraint)
(optf,optx,ret) = optimize!(opt,[0.25,0.25,0.25,0.25]) 

base = prod(1 .+  ite ./ 100)

rety = base ^ (252/size(df,1)) - 1

for i in 1:4
    @printf "%.1f%s for %s\n" optx[i]*100 '%' name[i]
end

global ite = zeros(size(df,1))

for i in 1:4
    global ite = ite .+ optx[i] .* df[:,name[i]]
end

@printf "annual return : %f\n" rety

cop1 = df[:,name[1]]
cop2 = df[:,name[2]]
cop3 = df[:,name[3]]
cop4 = df[:,name[4]]

tenanr1 = cumprod(1 .+ cop1 ./ 100) .- 1
tenanr2 = cumprod(1 .+ cop2 ./ 100) .- 1
tenanr3 = cumprod(1 .+ cop3 ./ 100) .- 1
tenanr4 = cumprod(1 .+ cop4 ./ 100) .- 1
tenanrp = cumprod(1 .+ ite ./ 100) .- 1

datecol =  DataFrame(CSV.File("D:\\Julia\\data.csv";select=["Date"]))
datearray = datecol[:,"Date"]

plot(datearray,tenanr1,label=name[1])
plot!(datearray,tenanr2,label=name[2])
plot!(datearray,tenanr3,label=name[3])
plot!(datearray,tenanr4,label=name[4])
plot!(datearray,tenanrp,label="portfolio")
plot!(xformatter = x -> Dates.format(Date(Dates.UTD(x)),"yyyy"))
savefig("test.png")

                
                     







