using PyPlot, PyCall
include("make_SQ_model.jl")
include("utilities.jl")

##### display plots to gui, or just save
## if they plot to gui, they don't have the size specified in teh code
pygui(true)

experiment_dict = Dict("n_vary"=>Dict(:inpts=>[:exit_distance],:epocs=>1000,:ex_locs=>[1.,5.],:cmp=>["ok","bad"],:legend_loc=>"upper right"),
                       "sense_vary"=>Dict(:inpts=>[:sensor_rwd],:epocs=>1000,:ex_locs=>[-35.,-150.],:cmp=>["ok","bad"],:legend_loc=>"lower right"),
                       "transition_vary"=>Dict(:inpts=>[:tprob],:epocs=>1000,:ex_locs=>[0.25,0.75],:cmp=>["ok","bad"],:legend_loc=>"lower right"),
                       "transition_e_vary"=>Dict(:inpts=>[:tprob,:e_mcts],:epocs=>1000,:ex_locs=>[],:cmp=>["ok","bad"],:legend_loc=>"lower right")
                      )
#  experiment_dict = Dict("transition_vary"=>Dict(:inpts=>[:tprob],:epocs=>1000,:ex_locs=>[0.25,0.75],:cmp=>["bad","ok"],:legend_loc=>"lower right"))
#  experiment_dict = Dict("n_vary"=>Dict(:inpts=>[:exit_distance],:epocs=>1000,:ex_locs=>[1.,5.],:cmp=>["bad"],:legend_loc=>"upper right"))
experiment_dict = Dict("transition_e_vary"=>Dict(:inpts=>[:tprob,:e_mcts],:epocs=>1000,:ex_locs=>[1.,5.],:cmp=>["ok","bad"],:legend_loc=>"upper right"))
#  experiment_dict = Dict("sense_vary"=>Dict(:inpts=>[:sensor_rwd],:epocs=>1000,:ex_locs=>[-35.,-150.],:cmp=>["bad"],:legend_loc=>"lower right"))

for expr in keys(experiment_dict)
    net_type = expr
    inpts = experiment_dict[expr][:inpts]
    epocs = experiment_dict[expr][:epocs]
    sq_example_locations = experiment_dict[expr][:ex_locs]
    cmp_list = experiment_dict[expr][:cmp]
    legend_loc = experiment_dict[expr][:legend_loc]
    println("Plotting: $(experiment_dict[expr])")
    for compare in cmp_list
        println("running '$compare' data")
        train_fname = "logs_old/$(net_type)_reference_solver_training.csv"
        test_fname = "logs_old/$(net_type)_$(compare)_solver.csv"

        inputs = Dict()
        for i in inpts
            inputs[i] = "ML.Continuous"
        end
        outputs = Dict(:X3_1=>"ML.Continuous",:X3_2=>"ML.Continuous")

        log_fname = "$(net_type)_$(make_label_from_keys(inputs))"
        log_loc = "nn_logs_old/"
        #  log_path = string(log_loc,"/",log_fname,"_net1-$(epocs)

        #  println("##########")
        #  println("$log_fname, $(readdir(log_loc))")
        #  println("##########")
        if !(any([contains(x,string(log_fname,"_")) for x in readdir(log_loc)]))
            println("No nn file exists, making one now")
            #  make_sq_model(net_type,inpts)
            make_sq_model(net_type,inpts,num_epoc=epocs)
        end

        param_files = searchdir(log_loc,log_fname,".params")

        num_epocs = parse(split(match(r"-\d+",param_files[1]).match,"-")[2])

        SQmodel = load_network(string(log_loc,log_fname),num_epocs,string(log_loc,log_fname,"_SQmodel.jld"))

        # get test and make predictions
        test_input, test_output, test_table, input_sch, output_sch = return_data(test_fname, inputs=inputs, outputs=outputs)

        data_mat = ML.featuremat(merge(input_sch,output_sch),test_table)

        info("restoring limits")
        limits = restore_eng_units(SQmodel.range,SQmodel.output_sch)
        info("getting predictions")
        _notused, pred_outputs = SQ_predict(SQmodel,test_input,use_eng_units=true)

        info("restoring test data")
        tst_in_eng = restore_eng_units(test_input,input_sch)
        tst_out_eng_ary = restore_eng_units(test_output,output_sch)

        # make figures
        if length(inputs) == 1
            fig,ax_ary = PyPlot.subplots(1,1,sharex=false)
            fig[:set_size_inches](8.0,6.0)
            fontsize = 20
            PyPlot.grid()

            i1 = collect(keys(inputs))[1]

            idx1 = nearest_to_x(tst_in_eng[i1],sq_example_locations[1])
            idx2 = nearest_to_x(tst_in_eng[i1],sq_example_locations[2])

            scatter_with_conf_bnds(ax_ary,tst_in_eng,tst_out_eng_ary,i1,:X3_1,:X3_2,:red,subsample=[idx1 idx2],label="candidate($(compare))",bar=true)
            #  scatter_with_conf_bnds(ax_ary,tst_in_eng,tst_out_eng_ary,i1,:X3_1,:X3_2,:red,subsample=collect(1:length(tst_in_eng[i1])),label="candidate",bar=true)
            #  ax_ary[:scatter](tst_in_eng[i1],tst_out_eng_ary[:X3_1],color=:red,alpha=0.2)

            ax_ary[:set_xlabel](string(i1),fontsize=fontsize)
            ax_ary[:set_ylabel](string("Reward"),fontsize=fontsize)
            ax_ary[:axhline](limits[:X3_1][2])
            ax_ary[:axhline](limits[:X3_1][1])

            add_sq_annotation(ax_ary,tst_in_eng,tst_out_eng_ary,pred_outputs,idx1,i1,:X3_1,:X3_2,SQmodel,fontsize=fontsize)
            add_sq_annotation(ax_ary,tst_in_eng,tst_out_eng_ary,pred_outputs,idx2,i1,:X3_1,:X3_2,SQmodel,fontsize=fontsize)

            ax_ary[:text](minimum(tst_in_eng[i1]),limits[:X3_1][2],L"r_H",fontsize=fontsize,va="bottom")
            ax_ary[:text](minimum(tst_in_eng[i1]),limits[:X3_1][1],L"r_L",fontsize=fontsize,va="top")

            #  ax_ary[:scatter](tst_in_eng[i1],pred_outputs[:X3_1],color=:blue)
            scatter_with_conf_bnds(ax_ary,tst_in_eng,pred_outputs,i1,:X3_1,:X3_2,:blue,subsample=collect(1:8:length(tst_in_eng[i1])),label="trusted",bar=false)
            ax_ary[:legend](loc=legend_loc,fontsize=fontsize)

            PyPlot.savefig(string("figs/",log_fname,"_",compare,".pdf"),dpi=300,transparent=true)
        elseif length(inputs) > 1
            # import stuff for plotting
            @pyimport mpl_toolkits.axes_grid1 as mpl_kit
            al = mpl_kit.make_axes_locatable

            function trust_surf(x,y,key)
                _notused, pred_outputs = SQ_predict(SQmodel,[x y]',use_eng_units=true)
                return pred_outputs[key][1]
            end

            i_rng = [-1.65;1.71]
            grid_ary = collect(linspace(i_rng[1],i_rng[2],50))
            ts = [trust_surf(x,y,:X3_1) for x in grid_ary, y in grid_ary]'
            tss = [trust_surf(x,y,:X3_2) for x in grid_ary, y in grid_ary]'

            display(ts)
            display(tss)

            # Solver Quality Surface
            fig = plt[:figure](figsize=(6.0,6.0))
            grid = fig[:subplots](2,2)
            div = []
            push!(div,al(grid[1]))
            push!(div,al(grid[2]))
            cax = []
            push!(cax,div[1][:append_axes]("right",size="5%",pad=0.05))
            push!(cax,div[2][:append_axes]("right",size="5%",pad=0.05))

            #  grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                                 #  nrows_ncols=(3,2),
                                 #  direction="row",
                                 #  aspect = true,
                                 #  axes_pad=0.65,
                                 #  share_all=true,
                                 #  cbar_location="right",
                                 #  cbar_mode="each",
                                 #  cbar_size="3%",
                                 #  cbar_pad=0.10,
                                 #  )
            fontsize = 10


            # make correlation plots
            i1 = inpts[1]
            i2 = inpts[2]

            #  corrplot(data_mat)
            poi1 = [0.2;200.] # point of interest, where X3 will be calculated
            poi2 = [0.7;800.] # point of interest, where X3 will be calculated
            poi1_norm = [(poi1[1]-mean(input_sch[i1]))/std(input_sch[i1]);(poi1[2]-mean(input_sch[i2]))/std(input_sch[i2])]
            poi2_norm = [(poi2[1]-mean(input_sch[i1]))/std(input_sch[i1]);(poi2[2]-mean(input_sch[i2]))/std(input_sch[i2])]

            x_rng = [i_rng[1]*std(input_sch[i1])+mean(input_sch[i1]);i_rng[2]*std(input_sch[i1])+mean(input_sch[i1])]
            y_rng = [i_rng[1]*std(input_sch[i2])+mean(input_sch[i2]);i_rng[2]*std(input_sch[i2])+mean(input_sch[i2])]
            subsample_num = 3

            aspct = diff(x_rng)[1]./diff(y_rng)[1]
            #  error()

            sq_cmap = :viridis
            im1 = grid[1][:imshow](ts,origin="lower",cmap=sq_cmap,interpolation="bilinear",extent=[x_rng;y_rng],aspect="auto")
            #  im1 = grid[1][:imshow](ts,origin="lower",cmap=sq_cmap,interpolation="bilinear",aspect="auto")
            grid[1][:scatter](poi1[1],poi1[2],color=:black,marker=L"$A$",color=:gray)
            grid[1][:scatter](poi2[1],poi2[2],color=:black,marker=L"$B$",color=:gray)
            grid[1][:set_ylabel]("$(inpts[2])")
            grid[1][:set_title]("Mean Rwd: Trusted Solver")
            fig[:colorbar](im1,cax=cax[1])

            sqs_cmap = :magma
            im2 = grid[2][:imshow](tss,origin="lower",cmap=sqs_cmap,interpolation="bilinear",extent=[x_rng;y_rng],aspect="auto")
            #  im2 = grid[2][:imshow](tss,origin="lower",cmap=sqs_cmap,interpolation="bilinear",aspect="auto")
            grid[2][:scatter](poi1[1],poi1[2],color=:black,s=50,marker=L"$A$",color=:gray)
            grid[2][:scatter](poi2[1],poi2[2],color=:black,s=50,marker=L"$B$",color=:gray)
            grid[2][:set_ylabel]("$(inpts[2])")
            grid[2][:set_xlabel]("$(inpts[1])")
            grid[2][:set_title]("Std: Trusted Solver")
            fig[:colorbar](im2,cax=cax[2])
            #  grid[2][:set_aspect](aspct)

            add_sq_scatter3d_annotation(grid[1],grid[3],test_input,tst_out_eng_ary,i1,i2,poi1_norm,SQmodel,marker=L"$A$",s=50,fontsize=12)
            add_sq_scatter3d_annotation(grid[2],grid[4],test_input,tst_out_eng_ary,i1,i2,poi2_norm,SQmodel,marker=L"$B$",s=50,fontsize=12)

            grid[3][:legend]()

            PyPlot.tight_layout()
            PyPlot.subplots_adjust(hspace=0.4,wspace=0.8)

            PyPlot.savefig(string("figs/",log_fname,"_",compare,".pdf"),dpi=300,transparent=true)
        else
         error("can't support more than 2 inputs yet")
        end
        PyPlot.close_figs()
    end
end

