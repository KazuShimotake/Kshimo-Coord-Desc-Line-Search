# Kshimo-Coord-Desc-Line-Search
Implementation of some versions of coordinate descent and line search, along with some interesting test functions.

You can see what the second two algorithms' outputs look like by commenting out cd_cyc_base,cd_rand_new in the opt_methods array in the display section

I ended up having to switch my first two algorithm's termination conditions to limited iterations rather than changes in x because my gss function is broken for some reason that I couldn't figure out and it was just running for several minutes and not actually improving, sometimes even jumping out to 1e+25 or more

However, my line search and gradient descent both actually get more efficient as the epsilon decreases. More specifically, it seems like they sharply increase in efficiency then flatten out. This would probably be easier to see if you cut out the two coordinate descent algorithms since they both throw off the scale by taking around 480,000 function calls in the 1,000 or so iterations I let them have.
