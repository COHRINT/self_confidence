function on_gcloud(;return_name::Bool=false,hostname::String="instance")
    # return if we're on google cloud, i.e. the word 'instance' is in the hostname
    if !("hostname" in readdir("/etc"))
        # if the "hostname" file doesn't exist we're not on gcloud
        return false
    end
    hname = open("/etc/hostname","r") do f
        readstring(f)
    end
    if return_name
        return hname
    end
    if contains(hname,hostname)
        return true
    else
        return false
    end
end
function send_mail(subject::String,body::String;to::String="brett.israelsen@gmail.com")
    # send an email via mutt, mutt has to be install and configured properly to do this.
    # this site might be helpful in doing that nickdesauliniers.github.io/blog/2016/06/18/mutt-gmail-ubuntu/
    run(pipeline(`echo $body`,`mutt -s $body $to`))
    println("sent mail")
end
