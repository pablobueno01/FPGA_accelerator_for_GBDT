-------------------------------------------------------------------------------
-- Generic data size register
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;

entity reg is
    generic(BITS: positive);
    port(-- Control signals
         Clk:   in std_logic;
         Reset: in std_logic;
         Load:  in std_logic;
         
         -- Input
         Din: in std_logic_vector(BITS - 1 downto 0);
         
         -- Output
         Dout: out std_logic_vector(BITS - 1 downto 0));
end reg;

architecture Behavioral of reg is
begin
process (Clk)
begin
    if rising_edge(Clk) then
        if Reset = '1' then
            Dout <= (others => '0');
        else
            if Load = '1' then
                Dout <= Din;
            end if;
        end if;
    end if;
end process;
end Behavioral;

