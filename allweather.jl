using CSV,DataFrames,StatsBase,Statistics,LinearAlgebra,NLopt
using Printf,Plots,Dates

#Input Data
df = DataFrame(CSV.File("D:\\Julia\\data.csv";select=["000852.SH","H11008.CSI","AU9999.SGE","CBA00301.CS"]))
datecol =  DataFrame(CSV.File("D:\\Julia\\data.csv";select=["Date"]))
rf = 0.03
#

#Obtaining objective weight
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
#

#Calculate Sharpe ratio,etc.
global ite = zeros(size(df,1))

for i in 1:4
    global ite = ite .+ optx[i] .* df[:,name[i]]
end

function yearet_cont(x::Vector)
    base = prod(1 .+  x ./ 100)
    rety = base ^ (252/size(df,1)) - 1
    return rety
end

cop1 = df[:,name[1]]
cop2 = df[:,name[2]]
cop3 = df[:,name[3]]
cop4 = df[:,name[4]]

function tenanr(x::Vector)
    tenanr = cumprod(1 .+ x ./ 100) .- 1
end

push!(name,"portfolio")

sharpe_ratio = zeros(1)
drawdown_rate= zeros(1)
drawdown_duration = zeros(1)

function sharpe_db_dur!(x::Vector,rf,s,dr,dd)
    sharpe = (yearet_cont(x) - rf)/(sqrt(12)*std(x ./ 100))
    
    hwm = zeros(1)
    drawdown = zeros(size(df,1))
    duration = zeros(size(df,1))

    tenanr = cumprod(1 .+ x ./ 100) .- 1
    
    for i in 2:(size(df,1))
        hwm[i-1]<tenanr[i] ? push!(hwm,tenanr[i]) : push!(hwm,hwm[i-1])
        drawdown[i] = (hwm[i] - tenanr[i])
        drawdown[i] == 0 ? duration[i] = 0 : duration[i] = duration[i-1] + 1
    end
    
    drawdown_max = maximum(drawdown)
    index = argmax(drawdown)
    hwm_max = hwm[index]
    duration_max = maximum(duration)

    if(s[1] == 0)
        s[1] = sharpe
        dr[1] = drawdown_max/hwm_max
        dd[1] = duration_max
    else
        push!(s,sharpe)
        push!(dr,drawdown_max/hwm_max)
        push!(dd,duration_max)
    end
    return nothing
end

sharpe_db_dur!(cop1,rf,sharpe_ratio,drawdown_rate,drawdown_duration)
sharpe_db_dur!(cop2,rf,sharpe_ratio,drawdown_rate,drawdown_duration)
sharpe_db_dur!(cop3,rf,sharpe_ratio,drawdown_rate,drawdown_duration)
sharpe_db_dur!(cop4,rf,sharpe_ratio,drawdown_rate,drawdown_duration)
sharpe_db_dur!(ite,rf,sharpe_ratio,drawdown_rate,drawdown_duration)

dfa = DataFrame(Asset=name,Sharpe = sharpe_ratio,Drawdown_rate = drawdown_rate,Drawdown_duration = drawdown_duration)
#

#Print the result
for i in 1:4
    @printf "%.1f%s for %s\n" optx[i]*100 '%' name[i]
end

@printf "annual return : %f\n" yearet_cont(ite)

@printf "Using risk free rate : %d%s\n" rf*100 '%'
display(dfa)
#

#Plot the ten year return
datearray = datecol[:,"Date"]

plot(datearray,tenanr(cop1),label=name[1])
plot!(datearray,tenanr(cop2),label=name[2])
plot!(datearray,tenanr(cop3),label=name[3])
plot!(datearray,tenanr(cop4),label=name[4])
plot!(datearray,tenanr(ite),label=name[5])
plot!(xformatter = x -> Dates.format(Date(Dates.UTD(x)),"yyyy"))
savefig("test.png")
#
