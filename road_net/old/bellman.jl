function bellman(mg::MetaGraph,output::Bool,its::Int,gamma::Float64)
    n = nv(mg)
    if output
        println("Output Enabled")
        print_func = println
    else
        println("No output")
        print_func = NULL
    end

    z = 0
    while z < its
        print_func("#######################")
        print_func("###### it-$(z+1) ##########")
        print_func("#######################")
        # println(z)
        z += 1
        Uprime_to_U!(mg)
        for i in vertices(mg) # for each state
            print_func("*****STATE $i*****")
            U = get_U_of_neighbors(mg,i,:U)
            Ui = mg.vprops[i][:U]
            # U_prime = get_U_of_neighbors(mg,i,:U_prime)
            # U = U_prime

            print_func("U: $Ui")
            # print_func("U_prime: $U_prime")

            R = mg.vprops[i][:reward]
            A = [mg.graph.fadjlist[i];i]

            print_func("Rwd: $R")

            s_primes = A' # reachable states -- neighbors and self
            sum_sprime = zeros(length(A))

            print_func("s_primes: $s_primes")
            print_func("----------")

            for j in 1:length(A) # for each action
                a = A[j]
                if output
                    print_func("**action $a **")
                end
                T = transition_prob(A,i,a,n)

                sum_sprime[j] = sum(T .* U) # this calculates over non-reachable states, this may need to be corrected in the future, for now it is fine

                print_func("T: $T")
                print_func("U: $U")
                print_func("sum term: $sum_sprime")
                if output
                    readline()
                end
            end

            max_a_idx = indmax(sum_sprime)
            max_a = A[max_a_idx]
            print_func("chosen a: $max_a")
            Up =  R + gamma*sum_sprime[max_a_idx]
            if i == 11 # fix utility for terminating states
                Up = 1.0
            elseif i == 10
                Up = -1.0
            end
            set_U!(mg,i,Up,:U_prime)
            print_func("U_prime set to: $(mg.vprops[i][:U_prime])")
        end
        if output
            print_func(get_all_U(mg))
            display("###### itr: $z #####")
            printU(get_all_U(mg))
            readline()
        end
    end
end

function Uprime_to_U!(mg::MetaGraph)
    # display(mg.vprops)
    for v in vertices(mg)
        mg.vprops[v][:U] = mg.vprops[v][:U_prime]
    end
end
function get_U_of_neighbors(mg::MetaGraph,v::Int,u_type::Symbol)
    n = neighbors(mg,v)
    U = zeros(length(n))
    for i = 1:length(n)
        U[i] = mg.vprops[n[i]][u_type]
    end
    return [U ;mg.vprops[v][u_type]]
end
function set_U!(mg::MetaGraph,v::Int,val::Float64,u_type::Symbol)
    mg.vprops[v][u_type] = val
end
function get_all_U(mg)
    U = zeros(nv(mg))
    for v in vertices(mg)
        U[v] = mg.vprops[v][:U]
    end
    return U
end
function printU(U::Array)
    Uany = convert(Array{Any},U)
    Uint = convert(Array{Any},collect(1:length(U)))
    insert!(Uany,5,"XXX")
    insert!(Uint,5,"XXX")
    display(reshape(Uint,3,4))
    display(reshape(Uany,3,4))

end

# transitions
function transition_prob(a_list::Array,s::Int,a::Int,n::Int)
    # a_list is the adjacency list of s
    # s is the current state
    # a is the proposed action
    # n is the total number of states
    # 80% chance of going in the desired direction, 20% chance of going a different direction
    T = zeros(n)
    for i in a_list
        if i == a
            T[i] = 0.8
        else
            T[i] = 0.2/(length(a_list)-1)
        end
    end
    # adj_list = length(a_list) - 1
    # if adj_list != 3
        # T[s] = (3-adj_list)*0.2/3
    # end
    @assert sum(T) == 1
    return T[a_list]
end
