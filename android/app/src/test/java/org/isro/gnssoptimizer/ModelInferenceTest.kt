package org.isro.gnssoptimizer

import org.isro.gnssoptimizer.model.FeatureEngineering
import org.isro.gnssoptimizer.model.RidgeModel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class ModelInferenceTest {

    private val temporalModel = RidgeModel.createTemporalModel()
    private val orientationModel = RidgeModel.createOrientationModel()

    private val temporalTestInputs = arrayOf(
        doubleArrayOf(4.967141530112e-01, -1.382643011712e-01, 6.476885381007e-01, 1.523029856408e+00, -2.341533747233e-01, -2.341369569492e-01, 1.579212815507e+00, 7.674347291529e-01, -4.694743859350e-01, 5.425600435860e-01, -4.634176928125e-01, -4.657297535703e-01, 2.419622715660e-01, -1.913280244658e+00, -1.724917832513e+00, -5.622875292410e-01, -1.012831120334e+00, 3.142473325953e-01, -9.080240755212e-01, -1.412303701335e+00, 1.465648768922e+00, -2.257763004865e-01, 6.752820468792e-02, -1.424748186213e+00, -5.443827245252e-01, 1.109225897099e-01, -1.150993577422e+00, 3.756980183457e-01, -6.006386899188e-01, -2.916937497933e-01, -6.017066122294e-01, 1.852278184509e+00, -1.349722473793e-02, -1.057710928956e+00, 8.225449121032e-01, -1.220843649971e+00),
        doubleArrayOf(2.088635950048e-01, -1.959670123880e+00, -1.328186048898e+00, 1.968612358691e-01, 7.384665799954e-01, 1.713682811900e-01, -1.156482823882e-01, -3.011036955893e-01, -1.478521990367e+00, -7.198442083947e-01, -4.606387709598e-01, 1.057122226219e+00, 3.436182895685e-01, -1.763040155363e+00, 3.240839693948e-01, -3.850822804163e-01, -6.769220003060e-01, 6.116762888409e-01, 1.030999522496e+00, 9.312801191162e-01, -8.392175232226e-01, -3.092123758512e-01, 3.312634314036e-01, 9.755451271224e-01, -4.791742378453e-01, -1.856589766638e-01, -1.106334974006e+00, -1.196206624081e+00, 8.125258223942e-01, 1.356240028571e+00, -7.201012158033e-02, 1.003532897892e+00, 3.616360250476e-01, -6.451197546051e-01, 3.613956055084e-01, 1.538036566466e+00),
        doubleArrayOf(-3.582603910995e-02, 1.564643655814e+00, -2.619745104090e+00, 8.219025043752e-01, 8.704706823817e-02, -2.990073504659e-01, 9.176077653550e-02, -1.987568914601e+00, -2.196718878375e-01, 3.571125715117e-01, 1.477894044742e+00, -5.182702182736e-01, -8.084936028932e-01, -5.017570435845e-01, 9.154021177021e-01, 3.287511096597e-01, -5.297602037670e-01, 5.132674331134e-01, 9.707754934804e-02, 9.686449905329e-01, -7.020530938774e-01, -3.276621465978e-01, -3.921081531322e-01, -1.463514948132e+00, 2.961202770646e-01, 2.610552721799e-01, 5.113456642461e-03, -2.345871333751e-01, -1.415370742050e+00, -4.206453227654e-01, -3.427145165268e-01, -8.022772692216e-01, -1.612857116660e-01, 4.040508568145e-01, 1.886185901211e+00, 1.745778128318e-01)
    )

    private val temporalExpectedOutputs = doubleArrayOf(
        2.278871929125e+01, 1.036750714601e+01, 1.816751696283e+01
    )

    private val orientationTestInputs = arrayOf(
        doubleArrayOf(2.575503907228e-01, -7.444591576617e-02, -1.918771215299e+00, -2.651387544922e-02, 6.023020994103e-02, 2.463242112485e+00, -1.923609647811e-01, 3.015473423336e-01, -3.471176970524e-02, -1.168678037620e+00, 1.142822814515e+00, 7.519330326868e-01, 7.910319470430e-01, -9.093874547947e-01, 1.402794310936e+00, -1.401851062792e+00, 5.868570938003e-01, 2.190455625810e+00, -9.905363251307e-01, -5.662977296028e-01),
        doubleArrayOf(9.965136508764e-02, -5.034756541162e-01, -1.550663431066e+00, 6.856297480603e-02, -1.062303713726e+00, 4.735924306352e-01, -9.194242342338e-01, 1.549934405018e+00, -7.832532923362e-01, -3.220615162057e-01, 8.135172173697e-01, -1.230864316434e+00, 2.274599346041e-01, 1.307142754282e+00, -1.607483234561e+00, 1.846338585323e-01, 2.598827942484e-01, 7.818228717773e-01, -1.236950710878e+00, -1.320456613084e+00),
        doubleArrayOf(5.219415656169e-01, 2.969846732332e-01, 2.504928503459e-01, 3.464482094970e-01, -6.800247215785e-01, 2.322536971610e-01, 2.930724732987e-01, -7.143514180264e-01, 1.865774511145e+00, 4.738329209118e-01, -1.191303497203e+00, 6.565536086338e-01, -9.746816702273e-01, 7.870846037425e-01, 1.158595579007e+00, -8.206823183517e-01, 9.633761292443e-01, 4.127809269365e-01, 8.220601599945e-01, 1.896792982654e+00)
    )

    private val orientationExpectedOutputs = doubleArrayOf(
        -7.150803081446e+01, 3.763646928370e+01, 4.686313829238e+01
    )

    @Test
    fun testTemporalModelInferenceEquivalence() {
        for (i in temporalTestInputs.indices) {
            val input = temporalTestInputs[i]
            val expected = temporalExpectedOutputs[i]
            val actual = temporalModel.predict(input)
            assertEquals("Temporal prediction $i mismatch", expected, actual, 1e-4)
        }
    }

    @Test
    fun testOrientationModelInferenceEquivalence() {
        for (i in orientationTestInputs.indices) {
            val input = orientationTestInputs[i]
            val expected = orientationExpectedOutputs[i]
            val actual = orientationModel.predict(input)
            assertEquals("Orientation prediction $i mismatch", expected, actual, 1e-4)
        }
    }

    @Test
    fun testCircularDelta() {
        assertEquals(20.0, FeatureEngineering.circularDeltaDeg(10.0, 350.0), 1e-6)
        assertEquals(-20.0, FeatureEngineering.circularDeltaDeg(350.0, 10.0), 1e-6)
        assertEquals(0.0, FeatureEngineering.circularDeltaDeg(180.0, 180.0), 1e-6)
        assertEquals(-180.0, FeatureEngineering.circularDeltaDeg(0.0, 180.0), 1e-6)
    }

    @Test
    fun testAirmassKY() {
        // At 90 deg (zenith), airmass should be close to 1.0
        val amZenith = FeatureEngineering.airmassKY(90.0)
        assertTrue("Zenith airmass should be close to 1.0", abs(amZenith - 1.0) < 0.05)
        
        // At 30 deg, sin(30) = 0.5, simple airmass = 2.0
        val am30 = FeatureEngineering.airmassKY(30.0)
        assertTrue("30 deg airmass should be ~2.0", abs(am30 - 2.0) < 0.1)
    }

    @Test
    fun testFeatureContributions() {
        val input = temporalTestInputs[0]
        val top = temporalModel.topContributions(input, 3)
        assertEquals(3, top.size)
        assertTrue("Rank 1 abs contribution should be >= Rank 2", top[0].absContribution >= top[1].absContribution)
        assertTrue("Rank 2 abs contribution should be >= Rank 3", top[1].absContribution >= top[2].absContribution)
    }
}
