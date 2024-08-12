-------------------------------------------------------------------------------
-- Comparator of 2 signals of 32 bits
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity comparator_2_32b is
    port(-- Control signals
         Clk:   in std_logic;
         Reset: in std_logic;
         Load:  in std_logic;
         
         -- Inputs to compare
         Input1: in std_logic_vector (31 downto 0);
         Input0: in std_logic_vector (31 downto 0);
         
         -- Output signals
         --     Greater:      the value of the selected input
         --     Bit_selected: the selected input
         Greater:      out std_logic_vector (31 downto 0);
         Bit_selected: out std_logic);
end comparator_2_32b;

architecture Behavioral of comparator_2_32b is
    
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
    
    -- (sel)ection_(reg) signals
    signal sel_reg_din, sel_reg_dout: std_logic_vector(0 downto 0);
    
    -- selected_(val)ue_(reg) signal
    signal val_reg_din: std_logic_vector(31 downto 0);

begin
    
    -- Comparation
    sel_reg_din <= "1" when (signed(Input1) >= signed(Input0)) else "0";
    val_reg_din <= Input1 when (sel_reg_din = "1") else Input0;
    
    -- Selection register
    selection_reg: reg
        generic map(BITS => 1)
        port map(Clk   => Clk,
                 Reset => Reset,
                 Load  => Load,
                 Din   => sel_reg_din,
                 Dout  => sel_reg_dout);
    
    -- Output mapping
    Bit_selected <= sel_reg_dout(0);
    
    -- Selected value register
    selected_value_reg: reg
        generic map(BITS => 32)
        port map(Clk   => Clk,
                 Reset => Reset,
                 Load  => Load,
                 Din   => val_reg_din,
                 Dout  => Greater);

end Behavioral;

