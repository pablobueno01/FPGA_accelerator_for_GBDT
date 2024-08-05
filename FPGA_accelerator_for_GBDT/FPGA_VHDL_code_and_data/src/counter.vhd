-------------------------------------------------------------------------------
-- Generic counter
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use work.types.all;

entity counter is
    generic(BITS: natural);
    port(-- Control signals
         Clk:   in std_logic;
         Reset: in std_logic;
         Count: in std_logic;
         
         -- Output
         Dout: out std_logic_vector(BITS - 1 downto 0));
end counter;

architecture Behavioral of counter is
    signal val: unsigned(BITS - 1 downto 0);
begin
    process (Clk, Reset)
    begin
        if rising_edge(Clk) then
            if Reset = '1' then
                val <= (others => '0');
            elsif Count = '1' then
                val <= val + 1;
            end if;
        end if;
    end process;
    Dout <= std_logic_vector(val);
end Behavioral;

