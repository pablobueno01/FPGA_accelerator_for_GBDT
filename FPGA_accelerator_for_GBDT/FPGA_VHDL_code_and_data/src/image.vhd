-------------------------------------------------------------------------------
-- Manager of an entire image with a number of classes <= 16 (3 threads)
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types.all;

entity image is
    generic(TREE_RAM_BITS: positive := 13;
            -- NUM_CLASSES:   positive := 16;   -- UNCOMMENT FOR IP
            -- NUM_FEATURES:  positive := 200); -- UNCOMMENT FOR IP
            NUM_CLASSES:   positive := 13;   -- UNCOMMENT FOR KSC
            NUM_FEATURES:  positive := 176); -- UNCOMMENT FOR KSC
            -- NUM_CLASSES:   positive := 9;   -- UNCOMMENT FOR PU
            -- NUM_FEATURES:  positive := 103); -- UNCOMMENT FOR PU
            -- NUM_CLASSES:   positive := 16;   -- UNCOMMENT FOR SV
            -- NUM_FEATURES:  positive := 224); -- UNCOMMENT FOR SV
    port(-- Control signals
         Clk:   in std_logic;
         Reset: in std_logic;
         
         -- Inputs for the nodes reception (trees)
        --  Load_trees: in std_logic;
        --  Valid_node: in std_logic;
        --  Addr:       in std_logic_vector(TREE_RAM_BITS - 1  downto 0);
        --  Trees_din:  in std_logic_vector(31 downto 0);
         
         -- Inputs for the features reception (pixels)
         Load_features: in std_logic;
         Valid_feature: in std_logic;
         Features_din:  in std_logic_vector(15 downto 0);
         Last_feature:  in std_logic;
         
         -- Output signals
         --     Finish:     finish (also 'ready') signal
         --     Dout:       the selected class
         --     Greater:    the value of the selected class prediction
         --     Curr_state: the current state
         Finish:     out std_logic;
         Dout:       out std_logic_vector(log_2(NUM_CLASSES) - 1 downto 0);
         Greater:    out std_logic_vector(31 downto 0);
         Curr_state: out std_logic_vector(2 downto 0));
end image;

architecture Behavioral of image is
    
    ---------------------------------------------------------------------------
    -- COMPONENTS
    ---------------------------------------------------------------------------
    
    component class is
        generic(TREE_RAM_BITS: positive;
                NUM_FEATURES:  positive);
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
    end component;
    
    component counter is
        generic(BITS: natural);
        port(-- Control signals
             Clk:   in std_logic;
             Reset: in std_logic;
             Count: in std_logic;
             
             -- Output
             Dout: out std_logic_vector(BITS - 1 downto 0));
    end component;
    
    component active_demux is
        generic(OUTPUT_LENGTH:  natural);
        port(Active: in  std_logic;
             Sel:    in  std_logic_vector (log_2(OUTPUT_LENGTH) - 1 downto 0);
             Dout:   out std_logic_vector (OUTPUT_LENGTH - 1 downto 0));
    end component;
    
    component reg is
        generic(BITS: positive);
        Port(-- Control signals
             Clk:   in std_logic;
             Reset: in std_logic;
             Load:  in std_logic;
             
             -- Input
             Din: in std_logic_vector(BITS - 1 downto 0);
             
             -- Output
             Dout: out std_logic_vector(BITS - 1 downto 0));
    end component;
    
    component comparator_16_32b is
        port(-- Control signals
             Clk:   in   std_logic;
             Reset: in   std_logic;
             Start: in   std_logic;
             
             -- Inputs to compare
             Input0:  in   std_logic_vector(31 downto 0);
             Input1:  in   std_logic_vector(31 downto 0);
             Input2:  in   std_logic_vector(31 downto 0);
             Input3:  in   std_logic_vector(31 downto 0);
             Input4:  in   std_logic_vector(31 downto 0);
             Input5:  in   std_logic_vector(31 downto 0);
             Input6:  in   std_logic_vector(31 downto 0);
             Input7:  in   std_logic_vector(31 downto 0);
             Input8:  in   std_logic_vector(31 downto 0);
             Input9:  in   std_logic_vector(31 downto 0);
             Input10: in   std_logic_vector(31 downto 0);
             Input11: in   std_logic_vector(31 downto 0);
             Input12: in   std_logic_vector(31 downto 0);
             Input13: in   std_logic_vector(31 downto 0);
             Input14: in   std_logic_vector(31 downto 0);
             Input15: in   std_logic_vector(31 downto 0);
             
             -- Output signals
             --     Done:      finish signal
             --     Greater:   the value of the selected input
             --     Selection: the selected input
             Done:      out  std_logic;
             Greater:   out  std_logic_vector(31 downto 0);
             Selection: out  std_logic_vector(3 downto 0));
    end component;
    
    ---------------------------------------------------------------------------
    -- STATES
    ---------------------------------------------------------------------------
    
    type SMC is (--S_IDLE, 
                S_TREES_LOADED, --S_LOAD_TREES,
                 S_LOAD_FEATURES, S_EXEC, S_ARGMAX);
    signal STATE, NEXT_STATE: SMC;
    
    ---------------------------------------------------------------------------
    -- SIGNALS
    ---------------------------------------------------------------------------
    
    -- (c)lass_(m)anager signals
    signal cm_start: std_logic;
    -- signal class_load: std_logic_vector(NUM_CLASSES - 1 downto 0);
    signal class_finish: std_logic_vector(NUM_CLASSES - 1 downto 0);
    -- To allow the number of classes to be a 'generic' value
    --
    --     TODO --> Argmax module is not generic!
    --     This initialization allows to use argmax module as a 'generic'
    --     module, no matter the number of classes.
    -- 
    signal class_dout: std_logic_vector(16 * 32 - 1 downto 0)
               := ((16 * 32 - 1) => '1', (15 * 32 - 1) => '1',
                   (14 * 32 - 1) => '1', (13 * 32 - 1) => '1',
                   (12 * 32 - 1) => '1', (11 * 32 - 1) => '1',
                   (10 * 32 - 1) => '1', (9 * 32 - 1)  => '1',
                   (8 * 32 - 1)  => '1', (7 * 32 - 1)  => '1',
                   (6 * 32 - 1)  => '1', (5 * 32 - 1)  => '1',
                   (4 * 32 - 1)  => '1', (3 * 32 - 1)  => '1',
                   (2 * 32 - 1)  => '1', (1 * 32 - 1)  => '1', others => '0');
    
    -- (c)lass_(c)ounter and (c)lass_(l)oader signals
    signal features: std_logic_vector(NUM_FEATURES * 16 - 1 downto 0);
    signal feature_load: std_logic_vector(NUM_FEATURES - 1 downto 0);
    -- signal cc_reset, cc_count, cl_active: std_logic;
    -- signal class_count: std_logic_vector(log_2(NUM_CLASSES) - 1 downto 0);
    
    -- (f)eature_(c)ounter and (f)eature_(l)oader signals
    signal fc_reset, fc_count, fl_active: std_logic;
    signal features_count: std_logic_vector(log_2(NUM_FEATURES) - 1 downto 0);
    
    -- (a)rgmax_(m)odule signals
    signal am_reset, am_start, am_done: std_logic;
    
    -- [METADATA] Only used when loading the trees
    -- signal last_class_node, last_node: std_logic;
    
    -- To allow the number of classes to be a 'generic' value
    -- 
    --     TODO --> Argmax module is not generic!
    --     'empty' signal could be used in a 'final implementation' to fill the
    --     argmax module extra inputs, avoiding the oversized 'class_dout'.
    -- 
    signal empty: std_logic_vector(31 downto 0) := (31 => '1', others => '0');
    signal selection_dout: std_logic_vector(3 downto 0);
    signal finish_state: std_logic_vector(NUM_CLASSES - 1 downto 0)
                             := (others => '1');

begin
    
    -- CLASSES
    ----------
    
    -- Class managers
    classes: for i in NUM_CLASSES - 1 downto 0 generate
        class_manager: class
            generic map(TREE_RAM_BITS => TREE_RAM_BITS,
                        NUM_FEATURES  => NUM_FEATURES)
            port map(Clk        => Clk ,
                     Reset      => Reset,
                     Start      => cm_start,
                    --  Load       => class_load(i),
                    --  Valid_data => Valid_node,
                    --  Addr       => Addr,
                    --  Ram_din    => Trees_din,
                     Features   => features,
                     Finish     => class_finish(i),
                     Dout       => class_dout((i + 1) * 32 - 1 downto i * 32));
    end generate;
    
    -- To keep the count of the classes when loading the trees
    -- class_counter: counter
    --     generic map(BITS => log_2(NUM_CLASSES))
    --     port map(Clk   => Clk, 
    --              Reset => cc_reset,
    --              Count => cc_count,
    --              Dout  => class_count);
    
    -- To select the class manager when loading the trees
    -- class_loader: active_demux
    --     generic map(OUTPUT_LENGTH  => NUM_CLASSES)
    --     port map(Active => cl_active,
    --              Sel    => class_count,
    --              Dout   => class_load);
    
    -- FEATURES
    -----------
    
    -- To store the current pixel
    regs: for i in NUM_FEATURES - 1 downto 0 generate
        feature_reg: reg
            generic map(BITS => 16)
            port map(Clk   => Clk,
                     Reset => Reset,
                     Load  => feature_load(i),
                     Din   => Features_din,
                     Dout  => features((i + 1) * 16 - 1 downto i * 16));
    end generate;
    
    -- To keep the count of the features when loading the pixel
    features_counter: counter
        generic map(BITS => log_2(NUM_FEATURES))
        port map(Clk   => Clk, 
                 Reset => fc_reset,
                 Count => fc_count,
                 Dout  => features_count);
    
    -- To select the feature register when loading the pixel
    features_loader: active_demux
        generic map(OUTPUT_LENGTH  => NUM_FEATURES)
        port map(Active => fl_active,
                 Sel    => features_count,
                 Dout   => feature_load);
    
    -- ARGMAX
    ---------
    
    -- To select the class with the higher accumulated prediction value
    argmax_module: comparator_16_32b 
        port map(Clk       => Clk,
                 Reset     => am_reset,
                 Start     => am_start,
                 Input0    => class_dout(1 * 32 - 1 downto 0 * 32),
                 Input1    => class_dout(2 * 32 - 1 downto 1 * 32),
                 Input2    => class_dout(3 * 32 - 1 downto 2 * 32),
                 Input3    => class_dout(4 * 32 - 1 downto 3 * 32),
                 Input4    => class_dout(5 * 32 - 1 downto 4 * 32),
                 Input5    => class_dout(6 * 32 - 1 downto 5 * 32),
                 Input6    => class_dout(7 * 32 - 1 downto 6 * 32),
                 Input7    => class_dout(8 * 32 - 1 downto 7 * 32),
                 Input8    => class_dout(9 * 32 - 1 downto 8 * 32),
                 Input9    => class_dout(10 * 32 - 1 downto 9 * 32),
                 Input10   => class_dout(11 * 32 - 1 downto 10 * 32),
                 Input11   => class_dout(12 * 32 - 1 downto 11 * 32),
                 Input12   => class_dout(13 * 32 - 1 downto 12 * 32),
                 Input13   => class_dout(14 * 32 - 1 downto 13 * 32),
                 Input14   => class_dout(15 * 32 - 1 downto 14 * 32),
                 Input15   => class_dout(16 * 32 - 1 downto 15 * 32),
                 Done      => am_done,
                 Greater   => Greater,
                 Selection => selection_dout);
    
    -- [METADATA] Only used when loading the trees
    -- last_class_node <= Trees_din(0) and Trees_din(1) and Trees_din(3);
    -- last_node       <= Trees_din(4);
    
    -- Final output
    Dout <= selection_dout(log_2(NUM_CLASSES) - 1 downto 0);
    
    -- PROCESSES
    ------------
    
    -- CLK process
    SM: process(Clk)
    begin
        if rising_edge(Clk) then
            if Reset = '1' then
                STATE <= S_TREES_LOADED;
            else
                STATE <= NEXT_STATE;
            end if;
        end if;
    end process;
    
    -- Main process
    SM_OUTPUT: process(STATE, Reset,
                       --Load_trees, 
                       Load_features,
                       Valid_feature, --Valid_node,
                       --last_class_node, last_node,
                       Last_feature, class_finish,
                       finish_state, am_done)
    begin
        
        -- Maintain the current state
        NEXT_STATE <= STATE;
        
        -- Signal to start the execution
        cm_start <= '0';
        
        -- Control signals of class_counter, feature_counter and argmax_module
        -- cc_count <= '0';
        fc_count <= '0';
        am_start <= '0';
        
        -- Reset signals of class_counter, feature_counter and argmax_module
        if Reset = '1' then
            -- cc_reset <= '1';
            fc_reset <= '1';
            am_reset <= '1';
        else
            -- cc_reset <= '0';
            fc_reset <= '0';
            am_reset <= '0';
        end if;
        
        -- Control signals of class_loader and feature_loader
        -- cl_active <= '0';
        fl_active <= '0';
        
        -- Finish signal
        Finish <= '0';
        
        case STATE is
            -- when S_IDLE =>
            --     Curr_state <= "000";
            --     if Load_trees = '1' then
            --         cl_active  <= '1';
            --         NEXT_STATE <= S_LOAD_TREES;
            --     end if;
            when S_TREES_LOADED =>
                Curr_state <= "010";
                Finish     <= '1';
                if Load_features = '1' then
                    if Valid_feature = '1' then
                        fl_active <= '1';
                        fc_count  <= '1';
                    end if;
                    NEXT_STATE <= S_LOAD_FEATURES;
                end if;
            -- when S_LOAD_TREES =>
            --     Curr_state <= "001";
            --     cl_active <= '1';
            --     if Valid_node = '1' and last_class_node = '1' then
            --         if last_node = '1' then
            --             cc_reset   <= '1';
            --             NEXT_STATE <= S_TREES_LOADED;
            --         else
            --             cc_count <= '1';
            --         end if;
            --     end if;
            when S_LOAD_FEATURES =>
                Curr_state <= "011";
                if Valid_feature = '1' then
                    fl_active <= '1';
                    fc_count  <= '1';
                    if Last_feature = '1' then
                        am_reset   <= '1';
                        cm_start   <= '1';
                        NEXT_STATE <= S_EXEC;
                    end if;
                end if;
            when S_EXEC =>
                Curr_state <= "100";
                if class_finish = finish_state then
                    NEXT_STATE <= S_ARGMAX;
                end if;
            when S_ARGMAX =>
                Curr_state <= "101";
                am_start <= '1';
                if am_done = '1' then
                    Finish     <= '1';
                    fc_reset   <= '1';
                    -- COMMENT FOR POWER MEASUREMENTS
                    NEXT_STATE <= S_TREES_LOADED;
                    -- NEXT_STATE <= S_EXEC; -- ONLY FOR POWER MEASUREMENTS
                    -- am_reset   <= '1';    -- ONLY FOR POWER MEASUREMENTS
                    -- cm_start   <= '1';    -- ONLY FOR POWER MEASUREMENTS
                end if;
            when OTHERS =>
                Curr_state <= "111";
        end case;
    end process;
end Behavioral;

