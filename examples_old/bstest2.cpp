

/**
 * Setup types
 */

Model 1: Bunch of blocks. 

Construction step:
- store a vector of virtual blocks
- blocks know their column and row by pointer



Con: Blockwise access is complex / difficult.

1. Get sparsity structure -> fills in the inner and outer indices
1a. Stores the offsets into valuePtr for each column (compressed or dense)

2. Update data -> directly to the 


/**
 * BSMatrix types
 */







BSMatrix

RowInfo
{
	Row

}


Column
-> unique ID
-> length

Row
-> unique ID
-> vector of blocks
-> length

Block
-> Column, Row


Row.add(new DenseBlock(Row, Column))
