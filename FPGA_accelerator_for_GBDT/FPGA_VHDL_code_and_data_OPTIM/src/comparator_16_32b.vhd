-------------------------------------------------------------------------------
-- Comparator of 16 signals of 32 bits
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity comparator_16_32b is
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
end comparator_16_32b;

architecture Behavioral of comparator_16_32b is
    
    component comparator_2_32b is
        port(-- Control signals
             Clk:   in std_logic;
             Reset: in std_logic;
             Load:  in std_logic;
             
             -- Inputs to compare
             Input1: in std_logic_vector(31 downto 0);
             Input0: in std_logic_vector(31 downto 0);
             
             -- Output signals
             --     Greater:      the value of the selected input
             --     Bit_selected: the selected input
             Greater:      out std_logic_vector(31 downto 0);
             Bit_selected: out std_logic);
    end component;
    
    component reg is
        Generic(BITS: positive);
        Port(-- Control signals
             Clk:   in std_logic;
             Reset: in std_logic;
             Load:  in std_logic;
             
             -- Input
             Din: in std_logic_vector(BITS - 1 downto 0);
             
             -- Output
             Dout: out std_logic_vector(BITS - 1 downto 0));
    end component;
    
    -- Level 0 signals
    signal greater0_level_0: std_logic_vector(31 downto 0);
    signal greater1_level_0: std_logic_vector(31 downto 0);
    signal greater2_level_0: std_logic_vector(31 downto 0);
    signal greater3_level_0: std_logic_vector(31 downto 0);
    signal greater4_level_0: std_logic_vector(31 downto 0);
    signal greater5_level_0: std_logic_vector(31 downto 0);
    signal greater6_level_0: std_logic_vector(31 downto 0);
    signal greater7_level_0: std_logic_vector(31 downto 0);
    signal bit_selected0_level_0: std_logic;
    signal bit_selected1_level_0: std_logic;
    signal bit_selected2_level_0: std_logic;
    signal bit_selected3_level_0: std_logic;
    signal bit_selected4_level_0: std_logic;
    signal bit_selected5_level_0: std_logic;
    signal bit_selected6_level_0: std_logic;
    signal bit_selected7_level_0: std_logic;
    
    -- Level 1 signals
    signal start_delay_1: std_logic_vector(0 downto 0);
    signal greater0_level_1: std_logic_vector(31 downto 0);
    signal greater1_level_1: std_logic_vector(31 downto 0);
    signal greater2_level_1: std_logic_vector(31 downto 0);
    signal greater3_level_1: std_logic_vector(31 downto 0);
    signal bit_selected0_level_1: std_logic;
    signal bit_selected1_level_1: std_logic;
    signal bit_selected2_level_1: std_logic;
    signal bit_selected3_level_1: std_logic;
    
    -- Level 2 signals
    signal start_delay_2: std_logic_vector(0 downto 0);
    signal greater0_level_2: std_logic_vector(31 downto 0);
    signal greater1_level_2: std_logic_vector(31 downto 0);
    signal bit_selected0_level_2: std_logic;
    signal bit_selected1_level_2: std_logic;
    
    -- Level 3 signals
    signal start_delay_3: std_logic_vector(0 downto 0);
    signal greater0_level_3: std_logic_vector(31 downto 0);
    signal bit_selected0_level_3: std_logic;
    
    -- Selection signals
    signal selection_0, selection_1, selection_2: std_logic;
    
    -- Control signals
    signal ctrl_0: std_logic_vector(2 downto 0);
    signal ctrl_1: std_logic_vector(1 downto 0);
    
    -- Done signals
    signal active_din: std_logic_vector(0 downto 0);
    signal fourth_cycle_dout: std_logic_vector(0 downto 0);

begin
    
    -- LEVEL 0
    
    first_level_0: comparator_2_32b
        port map(Clk          => Clk,
                 Reset        => Reset,
                 Load         => Start,
                 Input1       => Input1,
                 Input0       => Input0,
                 Greater      => greater0_level_0,
                 bit_selected => bit_selected0_level_0);
    
    first_level_1: comparator_2_32b
        port map(Clk          => Clk,
                 Reset        => Reset,
                 Load         => Start,
                 Input1       => Input3,
                 Input0       => Input2,
                 Greater      => greater1_level_0,
                 bit_selected => bit_selected1_level_0);
    
    first_level_2: comparator_2_32b
        port map(Clk          => Clk,
                 Reset        => Reset,
                 Load         => Start,
                 Input1       => Input5,
                 Input0       => Input4,
                 Greater      => greater2_level_0,
                 bit_selected => bit_selected2_level_0);
    
    first_level_3: comparator_2_32b
        port map(Clk          => Clk,
                 Reset        => Reset,
                 Load         => Start,
                 Input1       => Input7,
                 Input0       => Input6,
                 Greater      => greater3_level_0,
                 bit_selected => bit_selected3_level_0);
    
    first_level_4: comparator_2_32b
        port map(Clk          => Clk,
                 Reset        => Reset,
                 Load         => Start,
                 Input1       => Input9,
                 Input0       => Input8,
                 Greater      => greater4_level_0,
                 bit_selected => bit_selected4_level_0);
    
    first_level_5: comparator_2_32b
        port map(Clk          => Clk,
                 Reset        => Reset,
                 Load         => Start,
                 Input1       => Input11,
                 Input0       => Input10,
                 Greater      => greater5_level_0,
                 bit_selected => bit_selected5_level_0);
    
    first_level_6: comparator_2_32b
        port map(Clk          => Clk,
                 Reset        => Reset,
                 Load         => Start,
                 Input1       => Input13,
                 Input0       => Input12,
                 Greater      => greater6_level_0,
                 bit_selected => bit_selected6_level_0);
    
    first_level_7: comparator_2_32b
        port map(Clk          => Clk,
                 Reset        => Reset,
                 Load         => Start,
                 Input1       => Input15,
                 Input0       => Input14,
                 Greater      => greater7_level_0,
                 bit_selected => bit_selected7_level_0);
    
    -- LEVEL 1
    
    second_level_0: comparator_2_32b
        port map(Clk          => Clk,
                 Reset        => Reset,
                 Load         => start_delay_1(0),
                 Input1       => greater1_level_0,
                 Input0       => greater0_level_0,
                 Greater      => greater0_level_1,
                 bit_selected => bit_selected0_level_1);
    
    second_level_1: comparator_2_32b
        port map(Clk          => Clk,
                 Reset        => Reset,
                 Load         => start_delay_1(0),
                 Input1       => greater3_level_0,
                 Input0       => greater2_level_0,
                 Greater      => greater1_level_1,
                 bit_selected => bit_selected1_level_1);
    
    second_level_2: comparator_2_32b
        port map(Clk          => Clk,
                 Reset        => Reset,
                 Load         => start_delay_1(0),
                 Input1       => greater5_level_0,
                 Input0       => greater4_level_0,
                 Greater      => greater2_level_1,
                 bit_selected => bit_selected2_level_1);
    
    second_level_3: comparator_2_32b
        port map(Clk          => Clk,
                 Reset        => Reset,
                 Load         => start_delay_1(0),
                 Input1       => greater7_level_0,
                 Input0       => greater6_level_0,
                 Greater      => greater3_level_1,
                 bit_selected => bit_selected3_level_1);
    
    -- LEVEL 2
    
    third_level_0: comparator_2_32b
        port map(Clk          => Clk,
                 Reset        => Reset,
                 Load         => start_delay_2(0),
                 Input1       => greater1_level_1,
                 Input0       => greater0_level_1,
                 Greater      => greater0_level_2,
                 bit_selected => bit_selected0_level_2);
    
    third_level_1: comparator_2_32b
        port map(Clk          => Clk,
                 Reset        => Reset,
                 Load         => start_delay_2(0),
                 Input1       => greater3_level_1,
                 Input0       => greater2_level_1,
                 Greater      => greater1_level_2,
                 bit_selected => bit_selected1_level_2);
    
    -- LEVEL 3
    
    fourth_level_0: comparator_2_32b
        port map(Clk          => Clk,
                 Reset        => Reset,
                 Load         => start_delay_3(0),
                 Input1       => greater1_level_2,
                 Input0       => greater0_level_2,
                 Greater      => greater0_level_3,
                 bit_selected => bit_selected0_level_3);
    
    -- GREATER
    
    Greater <= greater0_level_3;
    
    -- SELECTION
    
    -- Selection bit 3
    Selection(3) <= bit_selected0_level_3;
    
    -- Selection bit 2
    selection_2  <= bit_selected1_level_2 when (bit_selected0_level_3 = '1')
                    else bit_selected0_level_2;
    Selection(2) <= selection_2;
    ctrl_1       <= bit_selected0_level_3 & selection_2;
    
    -- Selection bit 1
    selection_1  <= bit_selected0_level_1 when ctrl_1="00" else
                    bit_selected1_level_1 when ctrl_1="01" else
                    bit_selected2_level_1 when ctrl_1="10" else
                    bit_selected3_level_1 when ctrl_1="11" else
                    '0';
    Selection(1) <= selection_1;
    ctrl_0       <= ctrl_1 & selection_1;
    
    -- Selection bit 0
    selection_0  <= bit_selected0_level_0 when ctrl_0="000" else
                    bit_selected1_level_0 when ctrl_0="001" else
                    bit_selected2_level_0 when ctrl_0="010" else
                    bit_selected3_level_0 when ctrl_0="011" else
                    bit_selected4_level_0 when ctrl_0="100" else
                    bit_selected5_level_0 when ctrl_0="101" else
                    bit_selected6_level_0 when ctrl_0="110" else
                    bit_selected7_level_0 when ctrl_0="111" else
                    '0';
    Selection(0) <= selection_0;
    
    -- DONE SIGNAL
    
    -- To load a "1"
    active_din <= "1";
    
    -- First cycle
    first_cycle: reg
        generic map(BITS => 1)
        port map(Clk   => Clk,
                 Reset => Reset,
                 Load  => Start,
                 Din   => active_din,
                 Dout  => start_delay_1);
    
    -- Second cycle
    second_cycle: reg
        generic map(BITS => 1)
        port map(Clk   => Clk,
                 Reset => Reset,
                 Load  => start_delay_1(0),
                 Din   => active_din,
                 Dout  => start_delay_2);
    
    -- Third cycle
    third_cycle: reg
        generic map(BITS => 1)
        port map(Clk   => Clk,
                 Reset => Reset,
                 Load  => start_delay_2(0),
                 Din   => active_din,
                 Dout  => start_delay_3);
    
    -- Fourth cycle
    fourth_cycle: reg
        generic map(BITS => 1)
        port map(Clk   => Clk,
                 Reset => Reset,
                 Load  => start_delay_3(0),
                 Din   => active_din,
                 Dout  => fourth_cycle_dout);
    
    -- Conversion between std_logic and std_logic_vector
    Done <= fourth_cycle_dout(0);

end Behavioral;

