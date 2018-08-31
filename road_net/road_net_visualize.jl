using Roadnet_MDP
using POMDPToolbox
using MCTS

function visualize(mdp::roadnet_with_pursuer,s::roadnet_pursuer_state,policy::MCTS.MCTSPlanner;ms::Int64=10,fldr::String="default_movie_folder",fname::String="frame",max_steps::Int64=1000)
    hist = HistoryRecorder(max_steps=max_steps)
    hist = simulate(hist, mdp, policy, s)

    ## check for directory
    if !(fldr in readdir())
        mkdir(fldr)
    end
    if !("images" in readdir(fldr))
        mkdir(joinpath(fldr,"images"))
    end

    img_path = string(joinpath(fldr,"images"))

    #clear out old files
    if !(isempty(readdir(img_path)))
        display("removing pre-existing files")
        run(`rm $img_path/$(readdir(img_path))`)
    end

    it = 1
    println("processing $(length(hist)) steps")
    for (s,a,sp) in eachstep(hist, "s,a,sp")
        #  a = action(policy,s)
        a_int = POMDPs.action_index(mdp,a)
        if POMDPs.action_index(mdp,a) <= length(neighbors(mdp.road_net,s.node))
            a_loc = neighbors(mdp.road_net,s.node)[a_int]
        else
            a_loc = empty!([1])
        end
        println("taking action: $a, from node $(s.node) to get to node: $(sp.node) pursuer going to: $(sp.pnode)")
        #  display_network(mdp.road_net,evader_locs=[s.node],pursuer_locs=[s.pnode],action_locs=[a_loc],fname=string(joinpath(img_path,fname),it))
        #  display_network(mdp.road_net,evader_locs=[s.node],pursuer_locs=[s.pnode],fname=string(joinpath(img_path,fname),prepend_zeros("$it",tot_chars=length(hist))))
        display_network(mdp.road_net,evader_locs=[s.node],pursuer_locs=[s.pnode],fname=string(joinpath(img_path,fname),prepend_zeros("$it",tot_chars=1)))

        it += 1
        #  return a
    end;
    # original command line:
    # for file in *.svg; do inkscape $file -z -d=400 -e ${file%svg}png; done
    # -d is dpi, -z means no gui, -e is export as png
    temp_log = mktemp()
    for file in searchdir(img_path,".svg")
        f_split = splitext(file)
        # pipeline function allows to pipe output of command to a temp file
        run(pipeline(`inkscape $(joinpath(img_path,file)) -d=400 -z -e $(joinpath(img_path,f_split[1])).png`, stdout= "$(temp_log[1])"));
    end
    #  MAKE SURE TO SORT THESE BEFORE MAKING VIDEO
    run(pipeline(`ffmpeg -r 0.5 -y -i $img_path/frame%d.png -crf 25 $fldr/movie.avi`, stdout="$(temp_log[1])",stderr="$(temp_log[1])"))
    display("wrote movie to $(fldr)/movie.avi")
end

searchdir(path,key) = filter(x->contains(x,key),readdir(path)) # found this function on a forum

function prepend_zeros(s::String;tot_chars::Int64=5)
    # Help keep the files in alphanumerical order
    current_length = length(s)
    num_zeros = tot_chars-current_length
    if num_zeros > 0
        zeros = repeat("0",tot_chars-current_length)
    else
        zeros = ""
    end
    return zeros*s
end

function print_hist(mdp::roadnet_with_pursuer,hist::MDPHistory)
    hist_log = ""
    it = 1
    for (s,a,r,sp) in eachstep(hist, "s,a,r,sp")
        #  a = action(policy,s)
        a_int = POMDPs.action_index(mdp,a)
        if POMDPs.action_index(mdp,a) <= length(neighbors(mdp.road_net,s.node))
            a_loc = neighbors(mdp.road_net,s.node)[a_int]
        else
            a_loc = empty!([1])
        end
        h_str = "$it --- taking action: $a, from node $(s.node) to get to node: $(sp.node) pursuer going to: $(sp.pnode) rwd: $r"
        hist_log = "$hist_log \n $h_str"
        it += 1
    end
    return "$hist_log \nreward: $(undiscounted_reward(hist)), discounted: $(discounted_reward(hist))"
end
