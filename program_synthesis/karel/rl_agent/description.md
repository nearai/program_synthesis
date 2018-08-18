# KAREL

Current implementation description of the karel environment.

#### Ideas:
+ Hindsight Experience Replay
+ Prioritized Experience Replay
+ Meta learning (MAML, Reptile)
+ Hierarchical methods

#### Reference:
+ [Deep RL in parametrized action space](https://arxiv.org/pdf/1511.04143.pdf)

## Karel Agent - Reinforcement Learning

Details of the Karel Agent intermediate structures.

### Task

Task: (input_grids, output_grids)
    Represent pair of IO that the program should satisfy

### State
    np.array(1, n)
    Padded sequence of tokens

### Action

Each action is of the form `ActionName, Parameters` where parameters is a tuple of value that fully characterize the action.
The parameters vary from action to action.

+ ADD_ACTION(new_location, karel_token)

+ REMOVE_ACTION(location,)

+ REPLACE_ACTION(location, karel_token)

+ UNWRAP_BLOCK(location,)

+ WRAP_BLOCK(block_type, cond_id, start, end)
    block_type: {if, while, repeat}

        if block_type is repeat:
            cond_id: Number of times to be repeated [2, 10]
        else:
            cond_id: Condition id

        start, end: Integers must belong to the same body.

+ WRAP_IFELSE(cond_id, if_start, else_start, end)

+ REPLACE_COND(location, cond_id) !!! This is not implemented yet

+ SWITCH_IF_WHILE(location,) !!! This is not implemented yet

##### Karel locations

    # points to token after insert point
    #           v    v                                      v
    # DEF run m( move IF c( markersPresent c) i( turnLeft i) m)
    #            0    1                                      2
    #           v    v    v
    # DEF run m( move move m)
    #            0    1    2

##### Karel tokens
+ move
+ turnLeft
+ turnRight
+ putMarker
+ pickMarker

##### Karel conditions
+ frontIsClear
+ leftIsClear
+ rightIsClear
+ markersPresent
+ noMarkersPresent
+ !frontIsClear
+ !leftIsClear
+ !rightIsClear

