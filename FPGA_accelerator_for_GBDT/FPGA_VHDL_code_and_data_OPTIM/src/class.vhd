-------------------------------------------------------------------------------
-- Tree stages manager for the trees of each class
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.types.all;

entity class is
    generic(TREE_RAM_BITS: positive;
            NUM_FEATURES:  positive;
            SELECT_ROM:    integer := 0);
    port(-- Generic control signals
         Clk:   in std_logic;
         Reset: in std_logic;
         Start: in std_logic;
         
         -- Inputs to load the tree structure
        --  Load:       in std_logic;
        --  Valid_data: in std_logic;
        --  Addr:       in std_logic_vector(TREE_RAM_BITS - 1  downto 0);
        --  Ram_din:    in std_logic_vector(31 downto 0);
         
         -- Consecutive features of the current pixel
         Features: in std_logic_vector(NUM_FEATURES * 16 - 1 downto 0);
         
         -- Output signals
         --     Finish: finish signal
         --     Dout:   accumulated prediction value
         Finish: out std_logic;
         Dout:   out std_logic_vector(31 downto 0));
end class;

architecture Behavioral of class is
    
    ---------------------------------------------------------------------------
    -- COMPONENTS
    ---------------------------------------------------------------------------
    
    component reg is
        generic(BITS: positive);
        port(-- Control signals
             Clk:   in std_logic;
             Reset: in std_logic;
             Load:  in std_logic;
             
             -- Input
             Din: in std_logic_vector(BITS - 1 downto 0);
             
             -- Output
             Dout: out std_logic_vector(BITS - 1 downto 0));
    end component;
    
    component rom is --component ram is
        generic(ADDRESS_BITS: positive := 16;
                DATA_LENGTH:  positive := 32;
                SELECT_ROM:   integer := 0);
        port(-- Control signals
             Clk: in std_logic;
             --We:  in std_logic;
             Re:  in std_logic;
             
             -- Input signals
             Addr: in std_logic_vector(ADDRESS_BITS - 1 downto 0);
             --Din:  in std_logic_vector(DATA_LENGTH - 1 downto 0);
             
             -- Output
             Dout: out std_logic_vector (DATA_LENGTH - 1 downto 0);
            initial_addr_2: out std_logic_vector (ADDRESS_BITS - 1 downto 0);
            initial_addr_3: out std_logic_vector (ADDRESS_BITS - 1 downto 0));
    end component;
    
    component mux is
        Generic(DATA_LENGTH: natural;
                NUM_INPUTS:  natural);
        Port(Ctrl: in std_logic_vector(log_2(NUM_INPUTS) - 1 downto 0);
             Din:  in std_logic_vector(NUM_INPUTS * DATA_LENGTH - 1 downto 0);
             Dout: out std_logic_vector(DATA_LENGTH - 1 downto 0));
    end component;
    
    ---------------------------------------------------------------------------
    -- STATES
    ---------------------------------------------------------------------------
    
    type SMC is (--S_IDLE, S_LOAD_1, S_LOAD_2, S_LOAD_3,
                 --S_GET_GROUP_2_ADDRESS, S_GET_GROUP_3_ADDRESS,
                 S_EXEC_FIRST, S_EXEC_SECOND,
                 S_TREES_LOADED, S_EXEC_1, S_EXEC_2, S_EXEC_3,
                 S_EXEC_1_ENDED, S_EXEC_2_ENDED, S_EXEC_3_ENDED);
    signal STATE, NEXT_STATE: SMC;
    
    ---------------------------------------------------------------------------
    -- SIGNALS
    ---------------------------------------------------------------------------
    
    -- Common curr_addr signal
    signal curr_addr: std_logic_vector(TREE_RAM_BITS - 1 downto 0);
    
    -- (c)urr_(a)ddr_reg_(1) control signals
    signal curr_addr_1: std_logic_vector(TREE_RAM_BITS - 1 downto 0);
    signal ca_1_load, ca_1_reset: std_logic;
    
    -- (i)nitial_(a)ddr_reg_(2) control signals
    signal initial_addr_2: std_logic_vector(TREE_RAM_BITS - 1 downto 0);
    -- signal ia_2_load, ia_2_reset: std_logic;
    
    -- (c)urr_(a)ddr_reg_(2) control signals
    signal ca_2_din: std_logic_vector(TREE_RAM_BITS - 1 downto 0);
    signal curr_addr_2: std_logic_vector(TREE_RAM_BITS - 1 downto 0);
    signal ca_2_load, ca_2_reset: std_logic;
    
    -- (i)nitial_(a)ddr_reg_(3) control signals
    signal initial_addr_3: std_logic_vector(TREE_RAM_BITS - 1 downto 0);
    -- signal ia_3_load, ia_3_reset: std_logic;
    
    -- (c)urr_(a)ddr_reg_(3) control signals
    signal ca_3_din: std_logic_vector(TREE_RAM_BITS - 1 downto 0);
    signal curr_addr_3: std_logic_vector(TREE_RAM_BITS - 1 downto 0);
    signal ca_3_load, ca_3_reset: std_logic;
    
    -- (l)ast_(a)ddr_reg_(1) control signals
    signal last_addr_1: std_logic_vector(TREE_RAM_BITS - 1 downto 0);
    signal la_1_load, la_1_reset: std_logic;
    
    -- (l)ast_(a)ddr_reg_(2) control signals
    signal last_addr_2: std_logic_vector(TREE_RAM_BITS - 1 downto 0);
    signal la_2_load, la_2_reset: std_logic;
    
    -- (t)rees_(d)ata_ram signals
    signal td_addr: std_logic_vector(TREE_RAM_BITS - 1 downto 0);
    -- signal td_din: std_logic_vector(31 downto 0);
    signal td_dout: std_logic_vector(21 downto 0);
    signal td_re: std_logic;--signal td_we, td_re: std_logic;
    
    -- (td)_dout_(r)eg signals
    signal tdr_dout: std_logic_vector(21 downto 0);
    signal tdr_load, tdr_reset: std_logic;
    
    -- [METADATA] Only used when loading the trees
    -- signal last_group_node: std_logic;
    
    -- (T)ree (N)ode data
    signal tn_feature: std_logic_vector(3 downto 0);
    signal tn_cmp_value: std_logic_vector(7 downto 0);
    signal tn_pred_value: std_logic_vector(6 downto 0);
    signal tn_next_tree: std_logic_vector(TREE_RAM_BITS - 1 downto 0);
    signal tn_next_node: std_logic_vector(TREE_RAM_BITS - 1 downto 0);
    signal tn_right_child, addr_jmp: std_logic_vector(6 downto 0);
    signal tn_is_leaf, tn_last_tree: std_logic;
    
    -- Selected feature
    signal Feature: std_logic_vector(15 downto 0);
    
    -- (f)eature_(r)eg signals
    signal fr_load, fr_reset: std_logic;
    signal fr_dout: std_logic_vector(15 downto 0);
    
    -- Comparator signals
    signal cmp_dout: std_logic;
    
    -- (res)ult register signals
    signal res_load, res_reset: std_logic;
    signal res_din, res_dout: std_logic_vector(31 downto 0);
    
    -- (f)inish_(g)roup_(r)eg signals
    signal fgr_reset, finish_group: std_logic;
    signal fgr_load_1, fgr_load_2, fgr_load_3: std_logic;
    signal fgr_din, finish_1, finish_2, finish_3: std_logic_vector(0 downto 0);

begin
    
    -- Register to update addr 1
    curr_addr_reg_1: reg
        generic map(BITS => TREE_RAM_BITS)
        port map(Clk   => Clk,
                 Reset => ca_1_reset,
                 Load  => ca_1_load,
                 Din   => curr_addr,
                 Dout  => curr_addr_1);
    
    -- Register to save initial addr 2
    -- initial_addr_reg_2: reg
    --     generic map(BITS => TREE_RAM_BITS)
    --     port map(Clk   => Clk,
    --              Reset => ia_2_reset,
    --              Load  => ia_2_load,
    --              Din   => Addr,
    --              Dout  => initial_addr_2);
    
    -- Register to update addr 2
    curr_addr_reg_2: reg
        generic map(BITS => TREE_RAM_BITS)
        port map(Clk   => Clk,
                 Reset => ca_2_reset,
                 Load  => ca_2_load,
                 Din   => ca_2_din,
                 Dout  => curr_addr_2);
    
    -- Register to save initial addr 3
    -- initial_addr_reg_3: reg
    --     generic map(BITS => TREE_RAM_BITS)
    --     port map(Clk   => Clk,
    --              Reset => ia_3_reset,
    --              Load  => ia_3_load,
    --              Din   => Addr,
    --              Dout  => initial_addr_3);
    
    -- Register to update addr 3
    curr_addr_reg_3: reg
        generic map(BITS => TREE_RAM_BITS)
        port map(Clk   => Clk,
                 Reset => ca_3_reset,
                 Load  => ca_3_load,
                 Din   => ca_3_din,
                 Dout  => curr_addr_3);
    
    -- Register to propagate the addr from stage 1 to stage 2
    last_addr_reg_1: reg
        generic map(BITS => TREE_RAM_BITS)
        port map(Clk   => Clk,
                 Reset => la_1_reset,
                 Load  => la_1_load,
                 Din   => td_addr,
                 Dout  => last_addr_1);
    
    -- Register to save the addr of stage 2 and calculate the @ of the children
    last_addr_reg_2: reg
        generic map(BITS => TREE_RAM_BITS)
        port map(Clk   => Clk,
                 Reset => la_2_reset,
                 Load  => la_2_load,
                 Din   => last_addr_1,
                 Dout  => last_addr_2);
    
    -- RAM where all the trees of the class are located
    trees_data_rom: rom --trees_data_ram: ram
        generic map(ADDRESS_BITS => TREE_RAM_BITS,
                    DATA_LENGTH => 22,
                    SELECT_ROM   => SELECT_ROM)
        port map(Clk  => Clk,
                --  We   => td_we,
                 Re   => td_re,
                 Addr => td_addr,
                --  Din  => td_din,
                 Dout => td_dout,
                 initial_addr_2 => initial_addr_2,
                 initial_addr_3 => initial_addr_3);
    
    -- RAM input
    -- td_din <= Ram_din;
    
    -- Readding is always possible
    td_re <= '1';
    
    -- [METADATA] Last node of the trees of a thread
    -- last_group_node <= td_din(0) and td_din(1) and td_din(2);
    
    -- To get the feature number on stage 2
    tn_feature <= td_dout(19 downto 16) when td_dout(0) = '0'
                  else (others => '0');
    
    -- Register to propagate the RAM output to stage 3
    td_dout_reg: reg
        generic map(BITS => 22)
        port map(Clk   => Clk,
                 Reset => tdr_reset,
                 Load  => tdr_load,
                 Din   => td_dout,
                 Dout  => tdr_dout);
    
    -- Non-leaf node fields
    tn_cmp_value   <= tdr_dout(15 downto 8);
    tn_right_child <= tdr_dout(7 downto 1);
    tn_is_leaf     <= tdr_dout(0);
    
    -- Leaf node fields
    tn_pred_value <= tdr_dout(21 downto 15);
    tn_next_tree  <= tdr_dout((TREE_RAM_BITS + 2) - 1  downto 2);
    tn_last_tree  <= tdr_dout(1);
    
    -- Mux to select the corresponding feature
    features_mux: mux
        generic map(DATA_LENGTH => 16,
                    NUM_INPUTS  => NUM_FEATURES)
        Port map(Ctrl => tn_feature(log_2(NUM_FEATURES) - 1 downto 0),
                 Din  => Features,
                 Dout => Feature);
    
    -- Register to propagate the selected feature to stage 3
    feature_reg: reg
        generic map(BITS => 16)
        port map(Clk   => Clk,
                 Reset => fr_reset,
                 Load  => fr_load,
                 Din   => Feature,
                 Dout  => fr_dout);
    
    -- Feature value comparation
    --cmp_dout <= '0' when (signed(fr_dout) <= signed(tn_cmp_value)) else '1';
    cmp_dout <= '0' when (unsigned(fr_dout) <= resize(unsigned(tn_cmp_value), fr_dout'length)) else '1';
    
    -- Mux to choose between the two children of the node
    --     left child  --> add 1 to the current address
    --     right child --> add 'tn_right_child' to the current address
    addr_jmp <= "0000001" when cmp_dout = '0' else tn_right_child;
    tn_next_node <= std_logic_vector(unsigned(last_addr_2)
                                     + unsigned(addr_jmp));
    
    -- Mux to select if we should address the next node or a new tree
    curr_addr <= tn_next_tree when tn_is_leaf = '1' else tn_next_node;
    
    -- Register to accumulate the result
    result: reg
        generic map(BITS => 32)
        port map(Clk   => Clk,
                 Reset => res_reset,
                 Load  => res_load,
                 Din   => res_din,
                 Dout  => res_dout);
    
    -- The content of the register is accumulated with the next prediction
    res_din <= std_logic_vector(signed(res_dout)
                                + resize(signed(tn_pred_value), 32));
    
    -- Registers to keep the finish signal of each thread
    finish_group_reg_1: reg
        generic map(BITS => 1)
        port map(Clk   => Clk,
                 Reset => fgr_reset,
                 Load  => fgr_load_1,
                 Din   => fgr_din,
                 Dout  => finish_1);
    
    finish_group_reg_2: reg
        generic map(BITS => 1)
        port map(Clk   => Clk,
                 Reset => fgr_reset,
                 Load  => fgr_load_2,
                 Din   => fgr_din,
                 Dout  => finish_2);
    
    finish_group_reg_3: reg
        generic map(BITS => 1)
        port map(Clk   => Clk,
                 Reset => fgr_reset,
                 Load  => fgr_load_3,
                 Din   => fgr_din,
                 Dout  => finish_3);
    
    -- To store a '1'
    fgr_din <= "1";
    
    -- Finish signal
    finish_group <= finish_1(0) and finish_2(0) and finish_3(0);
    
    -- Outputs
    Finish <= finish_group;
    Dout   <= res_dout;
    
    -- PROCESS
    ----------
    
    -- CLK
    SM: process(Clk)
    begin
        if rising_edge(Clk) then
            if Reset = '1' then
                STATE <= S_TREES_LOADED; --STATE <= S_IDLE;
            else
                STATE <= NEXT_STATE;
            end if;
        end if;
    end process;
    
    -- Main process
    SM_OUTPUT: process(STATE, --Addr, 
                        curr_addr, Reset,
                       --Load, Valid_data, 
                       Start,
                       curr_addr_1, curr_addr_2, curr_addr_3,
                       initial_addr_2, initial_addr_3,
                       --last_group_node, 
                       tn_is_leaf,
                       tn_last_tree, finish_group,
                       finish_1, finish_2, finish_3)
    begin
        
        -- To keep current state
        NEXT_STATE <= STATE;
        
        -- RAM address
        -- td_we   <= '0';
        td_addr <= curr_addr_1;-- td_addr <= Addr;
        
        -- 'addr Registers' address
        ca_2_din <= curr_addr;
        ca_3_din <= curr_addr;
        
        -- Load signals
        la_1_load   <= '1';
        la_2_load   <= '1';
        fr_load     <= '1';
        tdr_load    <= '1';
        ca_1_load   <= '0';
        -- ia_2_load   <= '0';
        ca_2_load   <= '0';
        -- ia_3_load   <= '0';
        ca_3_load   <= '0';
        res_load    <= '0';
        fgr_load_1  <= '0';
        fgr_load_2  <= '0';
        fgr_load_3  <= '0';
        
        -- Reset signals
        if Reset = '1' then
            fr_reset   <= '1';
            tdr_reset  <= '1';
            la_1_reset <= '1';
            la_2_reset <= '1';
            ca_1_reset <= '1';
            -- ia_2_reset <= '1';
            ca_2_reset <= '1';
            -- ia_3_reset <= '1';
            ca_3_reset <= '1';
            res_reset  <= '1';
            fgr_reset  <= '1';
        else
            fr_reset   <= '0';
            tdr_reset  <= '0';
            la_1_reset <= '0';
            la_2_reset <= '0';
            ca_1_reset <= '0';
            -- ia_2_reset <= '0';
            ca_2_reset <= '0';
            -- ia_3_reset <= '0';
            ca_3_reset <= '0';
            res_reset  <= '0';
            fgr_reset  <= '0';
        end if;
        
        case STATE is
            -- when S_IDLE =>
            --     if Load = '1' then
            --         td_we      <= Valid_data;
            --         NEXT_STATE <= S_LOAD_1;
            --     end if;
            -- when S_LOAD_1 =>
            --     td_we <= Valid_data;
            --     if Valid_data = '1' and last_group_node = '1' then
            --         NEXT_STATE <= S_GET_GROUP_2_ADDRESS;
            --     end if;
            -- when S_GET_GROUP_2_ADDRESS =>
            --     td_we <= Valid_data;
            --     if Valid_data = '1' then
            --         ia_2_load  <= '1';
            --         NEXT_STATE <= S_LOAD_2;
            --     end if;
            -- when S_LOAD_2 =>
            --     td_we <= Valid_data;
            --     if Valid_data = '1' and last_group_node = '1' then
            --         NEXT_STATE <= S_GET_GROUP_3_ADDRESS;
            --     end if;
            -- when S_GET_GROUP_3_ADDRESS =>
            --     td_we <= Valid_data;
            --     if Valid_data = '1' then
            --         ia_3_load  <= '1';
            --         NEXT_STATE <= S_LOAD_3;
            --     end if;
            -- when S_LOAD_3 =>
            --     td_we <= Valid_data;
            --     if Valid_data = '1' and last_group_node = '1' then
            --         NEXT_STATE <= S_TREES_LOADED;
            --     end if;
            when S_TREES_LOADED =>
                td_addr <= curr_addr_1;
                if Start = '1' then
                    ca_2_din   <= initial_addr_2;
                    ca_2_load  <= '1';
                    ca_3_din   <= initial_addr_3;
                    ca_3_load  <= '1';
                    ca_1_reset <= '1';
                    la_1_reset <= '1';
                    res_reset  <= '1';
                    fgr_reset  <= '1';
                    NEXT_STATE <= S_EXEC_FIRST;
                end if;
            when S_EXEC_FIRST =>
                td_addr    <= curr_addr_1; -- (2) | 1 | (3) | (2)
                NEXT_STATE <= S_EXEC_SECOND;
            when S_EXEC_SECOND =>
                td_addr    <= curr_addr_2; -- (3) | 2 | 1 | (3)
                NEXT_STATE <= S_EXEC_3;
            when S_EXEC_1 =>
                td_addr   <= curr_addr_1; -- (2) | 1 | 3 | 2
                ca_2_load <= '1';         -- Load @ of thread 2
                if tn_is_leaf = '1' then  -- Test thread 2
                    -- Leaf node of thread 2
                    res_load <= '1';
                    if tn_last_tree = '1' then
                        fgr_load_2 <= '1';
                    end if;
                end if;
                if finish_group = '1' then
                    NEXT_STATE <= S_TREES_LOADED;
                else
                    if finish_3 = "1" then
                        NEXT_STATE <= S_EXEC_2_ENDED;
                    else
                        NEXT_STATE <= S_EXEC_2;
                    end if;
                end if;
            when S_EXEC_2 =>
                td_addr   <= curr_addr_2; -- (3) | 2 | 1 | 3
                ca_3_load <= '1';         -- Load @ of thread 3
                if tn_is_leaf = '1' then  -- Test thread 3
                    -- Leaf node of thread 3
                    res_load <= '1';
                    if tn_last_tree = '1' then
                        fgr_load_3 <= '1';
                    end if;
                end if;
                if finish_group = '1' then
                    NEXT_STATE <= S_TREES_LOADED;
                else
                    if finish_1 = "1" then
                        NEXT_STATE <= S_EXEC_3_ENDED;
                    else
                        NEXT_STATE <= S_EXEC_3;
                    end if;
                end if;
            when S_EXEC_3 =>
                td_addr   <= curr_addr_3; -- (1) | 3 | 2 | 1
                ca_1_load <= '1';         -- Load @ of thread 1
                if tn_is_leaf = '1' then  -- Test thread 1
                    -- Leaf node of thread 1
                    res_load <= '1';
                    if tn_last_tree = '1' then
                        fgr_load_1 <= '1';
                    end if;
                end if;
                if finish_group = '1' then
                    NEXT_STATE <= S_TREES_LOADED;
                else
                    if finish_2 = "1" then
                        NEXT_STATE <= S_EXEC_1_ENDED;
                    else
                        NEXT_STATE <= S_EXEC_1;
                    end if;
                end if;
             when S_EXEC_1_ENDED =>
                td_addr    <= curr_addr_1; -- (2) | 1 | 3 | END 2
                ca_2_load  <= '1';         -- Load @ of thread 2
                if finish_group = '1' then
                    NEXT_STATE <= S_TREES_LOADED;
                else
                    if finish_3 = "1" then
                        NEXT_STATE <= S_EXEC_2_ENDED;
                    else
                        NEXT_STATE <= S_EXEC_2;
                    end if;
                end if;
            when S_EXEC_2_ENDED =>
                td_addr    <= curr_addr_2; -- (3) | 2 | 1 | END 3
                ca_3_load  <= '1';         -- Load @ of thread 3
                if finish_group = '1' then
                    NEXT_STATE <= S_TREES_LOADED;
                else
                    if finish_1 = "1" then
                        NEXT_STATE <= S_EXEC_3_ENDED;
                    else
                        NEXT_STATE <= S_EXEC_3;
                    end if;
                end if;
            when S_EXEC_3_ENDED =>
                td_addr    <= curr_addr_3; -- (1) | 3 | 2 | END 1
                ca_1_load  <= '1';         -- Load @ of thread 1
                if finish_group = '1' then
                    NEXT_STATE <= S_TREES_LOADED;
                else
                    if finish_2 = "1" then
                        NEXT_STATE <= S_EXEC_1_ENDED;
                    else
                        NEXT_STATE <= S_EXEC_1;
                    end if;
                end if;
            when OTHERS =>
        end case;
    end process;
    
end Behavioral;

